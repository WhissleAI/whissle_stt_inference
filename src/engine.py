import json
import logging
import re
import threading
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .preprocessor import MelSpectrogramPreprocessor
from .decoder import CTCGreedyDecoder, CTCBeamSearchDecoder, _is_metadata_token

logger = logging.getLogger(__name__)


TOKENIZER_GROUPS = {
    "ENGLISH": ["en"],
}

_LANG_TO_GROUP = {}
for _g, _codes in TOKENIZER_GROUPS.items():
    for _c in _codes:
        _LANG_TO_GROUP[_c] = _g

_WB_BIAS_PER_GROUP: Dict[str, float] = {"ENGLISH": 3.0}
_WB_BIAS_DEFAULT = 0.0


class ASREngine:
    """
    ONNX-based ASR inference engine.
    Loads an ONNX model + optional KenLM LM and performs CTC decoding
    with metadata/entity extraction. Supports dual-head tag classifiers.
    """

    METADATA_PATTERNS = {
        'age': re.compile(r'\bAGE_[<>]?\w+\b'),
        'gender': re.compile(r'\bGENDER_\w+\b'),
        'emotion': re.compile(r'\bEMOTION_\w+\b|\b(?:NEUTRAL|SAD|DISGUST|HAPPY|FEAR|SURPRISE|ANGRY)\b'),
        'intent': re.compile(r'\bINTENT[-_]\w+\b'),
        'dialect': re.compile(r'\bDIALECT_\w+\b'),
    }

    ENTITY_PATTERN = re.compile(
        r'ENTITY[-_](\w+)\s+(.+?)'
        r'(?:\s+END\b'
        r'|(?=\s*\.?\s*(?:'
            r'INTENT[-_]'
            r'|ENTITY[-_]'
            r'|(?:NEUTRAL|SAD|DISGUST|HAPPY|FEAR|SURPRISE|ANGRY)\b'
            r'|AGE_|GENDER_|EMOTION_'
        r'))'
        r'|(?=\s*\.\s*$)'
        r'|\s*$)'
    )

    def __init__(
        self,
        model_dir: str,
        device: str = 'cpu',
        silence_padding: float = 0.3,
        lm_dir: Optional[str] = None,
        default_language: str = 'en',
        beam_width: int = 100,
        lm_alpha: float = 0.1,
        lm_beta: float = 0.5,
    ):
        self.model_dir = Path(model_dir)
        self.silence_padding = silence_padding
        self.default_language = default_language

        config_path = self.model_dir / 'config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.preprocessor = MelSpectrogramPreprocessor(self.config['preprocessor'])
        self.sample_rate = self.config['preprocessor'].get('sample_rate', 16000)

        vocab_path = self.model_dir / 'vocabulary.json'
        self._greedy_decoder = CTCGreedyDecoder(str(vocab_path), str(self.model_dir))

        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        self.vocabulary = vocab_data.get('vocabulary', [])

        self._lm_dir = Path(lm_dir) if lm_dir else self.model_dir / 'kenlm'
        self._beam_decoders: Dict[str, CTCBeamSearchDecoder] = {}
        self._beam_fallback: Optional[CTCBeamSearchDecoder] = None
        self._beam_width = beam_width
        self._lm_alpha = lm_alpha
        self._lm_beta = lm_beta
        self._vocab_path = str(vocab_path)
        self._tokenizer_dir = str(self.model_dir)

        self._init_beam_decoders()
        self.decoder = self._select_decoder(self.default_language)
        self._load_onnx_model(device)

        self._decoder_locks: Dict[int, threading.Lock] = {}
        self._decoder_locks_guard = threading.Lock()

        self._tag_classifier = self._load_tag_classifier()

    def _load_tag_classifier(self) -> Optional[Dict[str, Any]]:
        cls_onnx = self.model_dir / 'tag_classifier.onnx'
        cls_meta = self.model_dir / 'tag_classifier.json'
        if not cls_onnx.exists() or not cls_meta.exists():
            return None

        import onnxruntime as ort

        with open(cls_meta) as f:
            meta = json.load(f)

        sess = ort.InferenceSession(
            str(cls_onnx),
            providers=['CPUExecutionProvider'],
        )

        raw_cats = meta.get('categories', {})
        label_maps = meta.get('label_maps', {})
        category_sizes = meta.get('category_sizes', {})

        if isinstance(raw_cats, list):
            categories = {}
            for cat in raw_cats:
                categories[cat] = {
                    'num_classes': category_sizes.get(cat, 0),
                    'labels': label_maps.get(cat, []),
                }
        elif isinstance(raw_cats, dict):
            categories = raw_cats
            for cat in categories:
                if 'labels' not in categories[cat] and cat in label_maps:
                    categories[cat]['labels'] = label_maps[cat]
        else:
            categories = {}

        cat_names = sorted(categories.keys())
        print(f"Tag classifier loaded: {cat_names}")
        for cat in cat_names:
            info = categories[cat]
            print(f"  {cat}: {info.get('num_classes', '?')} classes — {info.get('labels', [])}")

        return {
            'session': sess,
            'meta': meta,
            'encoder_dim': meta.get('encoder_dim'),
            'categories': categories,
        }

    def _run_tag_classifier(self, encoder_output: np.ndarray,
                            length: Optional[np.ndarray] = None) -> Dict[str, Optional[str]]:
        if self._tag_classifier is None:
            return {}

        enc_dim = self._tag_classifier.get('encoder_dim', 1024)
        if encoder_output.ndim == 3:
            if encoder_output.shape[1] == enc_dim and encoder_output.shape[2] != enc_dim:
                encoder_output = np.transpose(encoder_output, (0, 2, 1))

        T = encoder_output.shape[1]
        if length is not None:
            valid_len = int(length[0])
            mask = np.zeros((1, T, 1), dtype=np.float32)
            mask[0, :valid_len, :] = 1.0
            pooled = (encoder_output * mask).sum(axis=1) / max(valid_len, 1)
        else:
            pooled = encoder_output.mean(axis=1)

        pooled = pooled.astype(np.float32)
        sess = self._tag_classifier['session']
        categories = self._tag_classifier['categories']
        outputs = sess.run(None, {'pooled_encoder': pooled})

        result = {}
        for i, cat in enumerate(sorted(categories.keys())):
            cat_info = categories[cat]
            logits = outputs[i][0]
            pred_idx = int(np.argmax(logits))
            labels = cat_info.get('labels', [])
            if pred_idx < len(labels):
                label = labels[pred_idx]
            else:
                label = f'{cat}_{pred_idx}'
            result[cat.lower()] = label if label != 'NONE' else None

        return result

    def _lock_for_decoder(self, dec: Any) -> threading.Lock:
        dec_id = id(dec)
        lock = self._decoder_locks.get(dec_id)
        if lock is None:
            with self._decoder_locks_guard:
                lock = self._decoder_locks.get(dec_id)
                if lock is None:
                    lock = threading.Lock()
                    self._decoder_locks[dec_id] = lock
        return lock

    def _wb_bias_for(self, group_or_lang: str) -> float:
        group = _LANG_TO_GROUP.get(group_or_lang, group_or_lang)
        return _WB_BIAS_PER_GROUP.get(group, _WB_BIAS_DEFAULT)

    def _init_beam_decoders(self) -> None:
        if not self._lm_dir.is_dir():
            print(f"No KenLM directory at {self._lm_dir} — using greedy decoder")
            self._try_build_fallback_beam()
            return

        all_known_codes = set()
        for codes in TOKENIZER_GROUPS.values():
            all_known_codes.update(codes)

        for lang in sorted(all_known_codes):
            lm_path = self._find_lm_file(lang)
            if lm_path is None:
                continue
            unigrams_path = self._lm_dir / f"{lang}.unigrams.txt"
            wb = self._wb_bias_for(lang)
            try:
                dec = CTCBeamSearchDecoder(
                    vocab_path=self._vocab_path,
                    tokenizer_dir=self._tokenizer_dir,
                    kenlm_model_path=str(lm_path),
                    unigrams_path=str(unigrams_path) if unigrams_path.exists() else None,
                    beam_width=self._beam_width,
                    alpha=self._lm_alpha,
                    beta=self._lm_beta,
                    word_boundary_bias=wb,
                )
                self._beam_decoders[lang] = dec
                print(f"KenLM beam decoder loaded for '{lang}' from {lm_path}")
            except Exception as exc:
                print(f"Warning: failed to load KenLM for '{lang}': {exc}")

        for group_name, lang_codes in TOKENIZER_GROUPS.items():
            lm_path = self._find_lm_file(group_name)
            if lm_path is None:
                continue
            unigrams_path = self._lm_dir / f"{group_name}.unigrams.txt"
            wb = self._wb_bias_for(group_name)
            try:
                dec = CTCBeamSearchDecoder(
                    vocab_path=self._vocab_path,
                    tokenizer_dir=self._tokenizer_dir,
                    kenlm_model_path=str(lm_path),
                    unigrams_path=str(unigrams_path) if unigrams_path.exists() else None,
                    beam_width=self._beam_width,
                    alpha=self._lm_alpha,
                    beta=self._lm_beta,
                    word_boundary_bias=wb,
                )
                self._beam_decoders[group_name] = dec
                for lc in lang_codes:
                    if lc not in self._beam_decoders:
                        self._beam_decoders[lc] = dec
                print(f"KenLM beam decoder loaded for group '{group_name}' from {lm_path}")
            except Exception as exc:
                print(f"Warning: failed to load KenLM for group '{group_name}': {exc}")

        if not self._beam_decoders:
            print("No KenLM models found — trying beam search without LM")
            self._try_build_fallback_beam()

    def _try_build_fallback_beam(self) -> None:
        try:
            self._beam_fallback = CTCBeamSearchDecoder(
                vocab_path=self._vocab_path,
                tokenizer_dir=self._tokenizer_dir,
                beam_width=self._beam_width,
                alpha=0.0,
                beta=0.0,
                word_boundary_bias=0.0,
            )
            print("Fallback beam search decoder ready (no LM)")
        except Exception as exc:
            print(f"Warning: beam search unavailable, using greedy: {exc}")

    def _find_lm_file(self, lang: str) -> Optional[Path]:
        for ext in (".bin", ".arpa.bin", ".arpa"):
            p = self._lm_dir / f"{lang}{ext}"
            if p.exists():
                return p
        return None

    def _select_decoder(self, language: Optional[str] = None):
        lang = (language or self.default_language).lower()
        if lang in self._beam_decoders:
            return self._beam_decoders[lang]
        group = _LANG_TO_GROUP.get(lang)
        if group and group in self._beam_decoders:
            return self._beam_decoders[group]
        if self._beam_fallback is not None:
            return self._beam_fallback
        return self._greedy_decoder

    @property
    def available_languages(self) -> List[str]:
        return sorted(k for k in self._beam_decoders if k not in TOKENIZER_GROUPS)

    @property
    def decoder_type(self) -> str:
        if self._beam_decoders:
            return "beam_search_kenlm"
        if self._beam_fallback is not None:
            return "beam_search"
        return "greedy"

    def _load_onnx_model(self, device: str) -> None:
        import onnxruntime as ort

        onnx_path = self.model_dir / 'model.onnx'
        available_providers = ort.get_available_providers()
        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"Available providers: {available_providers}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        use_cuda = device in ('auto', 'cuda') and 'CUDAExecutionProvider' in available_providers
        if use_cuda:
            try:
                self.session = ort.InferenceSession(
                    str(onnx_path),
                    sess_options=sess_options,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                )
            except Exception as e:
                print(f"CUDA failed: {e}, falling back to CPU")
                self.session = ort.InferenceSession(
                    str(onnx_path),
                    sess_options=sess_options,
                    providers=['CPUExecutionProvider'],
                )
        else:
            self.session = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_options,
                providers=['CPUExecutionProvider'],
            )

        print(f"ONNX active providers: {self.session.get_providers()}")

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.has_length_input = len(self.session.get_inputs()) > 1
        if self.has_length_input:
            self.length_input_name = self.session.get_inputs()[1].name

        onnx_outputs = self.session.get_outputs()
        self._encoder_output_name = None
        self._encoded_lengths_name = None
        if len(onnx_outputs) > 1:
            for out in onnx_outputs[1:]:
                if 'encoder' in out.name.lower():
                    self._encoder_output_name = out.name
                    print(f"Dual-head model: encoder output '{out.name}'")
                elif 'length' in out.name.lower():
                    self._encoded_lengths_name = out.name

        # Detect logprobs format: [B, C, T] vs [B, T, C]
        logprobs_shape = onnx_outputs[0].shape
        self._logprobs_channel_first = False
        if len(logprobs_shape) == 3:
            dim1 = logprobs_shape[1]
            if isinstance(dim1, int) and dim1 == len(self.vocabulary) + 1:
                self._logprobs_channel_first = True
                print(f"Logprobs format: [B, C={dim1}, T] (channel-first, will transpose)")

    def _prepare_onnx_inputs(self, audio: np.ndarray, sample_rate: Optional[int]) -> Dict[str, np.ndarray]:
        mel_spec = self.preprocessor(audio, sample_rate)
        mel_spec = np.expand_dims(mel_spec, axis=0)
        if self.has_length_input:
            length = np.array([mel_spec.shape[2]], dtype=np.int64)
            return {self.input_name: mel_spec, self.length_input_name: length}
        return {self.input_name: mel_spec}

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        import librosa
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        return audio.astype(np.float32), sr

    def load_audio_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        import librosa
        import soundfile as sf
        try:
            audio, sr = sf.read(BytesIO(audio_bytes))
        except Exception:
            audio, sr = librosa.load(BytesIO(audio_bytes), sr=None, mono=True)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), sr

    def _add_silence_padding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.silence_padding > 0:
            silence_samples = int(self.silence_padding * sample_rate)
            silence = np.zeros(silence_samples, dtype=audio.dtype)
            audio = np.concatenate([silence, audio, silence])
        return audio

    def _mask_silence_logits(
        self, logits: np.ndarray, blank_id: int, padded_audio_len: int
    ) -> np.ndarray:
        if self.silence_padding <= 0 or padded_audio_len <= 0:
            return logits

        total_frames = logits.shape[0]
        padding_samples = int(self.silence_padding * self.sample_rate)
        silence_frames = max(1, round(padding_samples / padded_audio_len * total_frames))
        mask_frames = silence_frames + 6

        if mask_frames >= total_frames:
            return logits

        logits = logits.copy()
        neg_val = float(np.finfo(logits.dtype).min) / 2
        logits[-mask_frames:, :] = neg_val
        if blank_id < logits.shape[1]:
            logits[-mask_frames:, blank_id] = 0.0

        return logits

    def _extract_metadata(self, text: str) -> Dict[str, Optional[str]]:
        metadata = {}
        for key, pattern in self.METADATA_PATTERNS.items():
            match = pattern.search(text)
            metadata[key] = match.group() if match else None
        return metadata

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        entities = []
        for match in self.ENTITY_PATTERN.finditer(text):
            entities.append({
                'type': match.group(1),
                'value': match.group(2).strip(),
                'raw': match.group(0)
            })
        return entities

    def _clean_transcript(self, text: str) -> str:
        text = self.ENTITY_PATTERN.sub(r'\2', text)
        for pattern in self.METADATA_PATTERNS.values():
            text = pattern.sub('', text)
        text = re.sub(r'\bEND\b', '', text)
        text = text.replace('<unk>', '')
        text = text.replace('|', ' ')
        text = ' '.join(text.split())
        return text.strip()

    @staticmethod
    def _safe_softmax(logits_1d: np.ndarray) -> Optional[np.ndarray]:
        fl = logits_1d - logits_1d.max()
        exp_fl = np.exp(fl)
        s = exp_fl.sum()
        if s <= 0 or not np.isfinite(s):
            return None
        return exp_fl / s

    def _get_metadata_probs(self, logits: np.ndarray, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        if logits is None or logits.ndim < 2 or logits.shape[0] == 0:
            return {}

        metadata_probs: Dict[str, List[Dict[str, Any]]] = {}

        for category, pattern in self.METADATA_PATTERNS.items():
            cat_indices: List[int] = []
            cat_tokens: List[str] = []
            for idx, token in enumerate(self.vocabulary):
                if pattern.match(token):
                    cat_indices.append(idx)
                    cat_tokens.append(token)

            if not cat_indices:
                continue

            cat_idx_arr = np.array(cat_indices)
            cat_logits = logits[:, cat_idx_arr]
            best_frame = int(np.max(cat_logits, axis=1).argmax())
            cat_probs = self._safe_softmax(cat_logits[best_frame])
            if cat_probs is None:
                continue

            category_probs = [
                {'token': tok, 'probability': float(cat_probs[i])}
                for i, tok in enumerate(cat_tokens)
            ]
            category_probs.sort(key=lambda x: x['probability'], reverse=True)

            if len(category_probs) > 50:
                category_probs = category_probs[:top_k]

            metadata_probs[category] = category_probs

        return metadata_probs

    def transcribe(
        self,
        audio: Union[str, bytes, np.ndarray],
        sample_rate: Optional[int] = None,
        metadata_prob: bool = False,
        top_k: int = 5,
        language: Optional[str] = None,
        use_lm: bool = True,
        beam_width: Optional[int] = None,
        lm_alpha: Optional[float] = None,
        lm_beta: Optional[float] = None,
        hotwords: Optional[List[str]] = None,
        hotword_weight: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text with metadata and entity extraction.

        Args:
            audio: Path to audio file, audio bytes, or numpy array
            sample_rate: Required if audio is numpy array
            metadata_prob: Include per-category probability distributions
            language: Language code for LM selection
            use_lm: Whether to use beam search + LM
            hotwords: Words/phrases to boost during beam search

        Returns:
            Dict with transcript, raw_output, metadata, entities, tags, etc.
        """
        import time
        t0 = time.perf_counter()

        if isinstance(audio, str):
            audio, sample_rate = self.load_audio(audio)
        elif isinstance(audio, bytes):
            audio, sample_rate = self.load_audio_bytes(audio)

        if isinstance(audio, np.ndarray) and audio.size == 0:
            return {'transcript': '', 'raw_output': '', 'metadata': {}, 'entities': [], 'tags': {}}

        audio_duration_sec = len(audio) / sample_rate if sample_rate else 0.0

        audio = self._add_silence_padding(audio, sample_rate)
        padded_audio_len = len(audio)

        inputs = self._prepare_onnx_inputs(audio, sample_rate)

        output_names = [self.output_name]
        if self._encoder_output_name:
            output_names.append(self._encoder_output_name)
        if self._encoded_lengths_name:
            output_names.append(self._encoded_lengths_name)

        onnx_outputs = self.session.run(output_names, inputs)
        logits = onnx_outputs[0]
        encoder_output = None
        encoded_lengths = None
        for i, name in enumerate(output_names[1:], 1):
            if name == self._encoder_output_name:
                encoder_output = onnx_outputs[i]
            elif name == self._encoded_lengths_name:
                encoded_lengths = onnx_outputs[i]

        if self._logprobs_channel_first:
            logits = np.transpose(logits, (0, 2, 1))
        logits = logits[0]

        greedy_output = self._greedy_decoder.decode(logits)

        blank_id = getattr(self._greedy_decoder, 'blank_id', len(self.vocabulary))
        logits_masked = self._mask_silence_logits(logits, blank_id, padded_audio_len)

        if use_lm:
            dec = self._select_decoder(language)
        else:
            dec = self._greedy_decoder

        if isinstance(dec, CTCBeamSearchDecoder):
            dec_lock = self._lock_for_decoder(dec)
            with dec_lock:
                prev_bw = None
                if beam_width is not None:
                    prev_bw = dec.beam_width
                    dec.beam_width = beam_width
                beam_output = dec.decode(
                    logits_masked,
                    hotwords=hotwords,
                    hotword_weight=hotword_weight,
                    alpha=lm_alpha,
                    beta=lm_beta,
                )
                if prev_bw is not None:
                    dec.beam_width = prev_bw
            clean_transcript = self._clean_transcript(beam_output)
        else:
            clean_transcript = self._clean_transcript(greedy_output)

        metadata = self._extract_metadata(greedy_output)
        entities = self._extract_entities(greedy_output)

        # Tag classifier results (dual-head model)
        tags = {}
        if encoder_output is not None:
            tags = self._run_tag_classifier(encoder_output, encoded_lengths)

        # Merge: tag classifier predictions override CTC-extracted metadata
        for cat, val in tags.items():
            if val is not None:
                metadata[cat] = val

        result: Dict[str, Any] = {
            'transcript': clean_transcript,
            'raw_output': greedy_output,
            'metadata': metadata,
            'entities': entities,
            'tags': tags,
            'duration': round(audio_duration_sec, 3),
            'processing_time': round(time.perf_counter() - t0, 3),
        }

        if metadata_prob:
            result['metadata_probs'] = self._get_metadata_probs(logits, top_k)

        return result

    def transcribe_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        return self.transcribe(file_path, **kwargs)

    def transcribe_bytes(self, audio_bytes: bytes, **kwargs) -> Dict[str, Any]:
        return self.transcribe(audio_bytes, **kwargs)

    def transcribe_pcm(
        self,
        pcm_data: bytes,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        dtype = np.int16 if sample_width == 2 else np.float32
        audio = np.frombuffer(pcm_data, dtype=dtype)
        if dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return self.transcribe(audio, sample_rate=sample_rate, **kwargs)
