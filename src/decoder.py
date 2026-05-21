import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class CTCGreedyDecoder:
    """CTC Greedy Decoder. Supports character, BPE, and aggregate tokenization."""

    def __init__(
        self,
        vocab_path: str,
        tokenizer_dir: Optional[str] = None,
        blank_id: Optional[int] = None
    ):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        self.vocabulary = vocab_data.get('vocabulary', vocab_data.get('labels', []))
        self.tokenizer_type = vocab_data.get('tokenizer_type', 'char')
        self.blank_id = blank_id if blank_id is not None else vocab_data.get(
            'blank_id', len(self.vocabulary)
        )

        self.is_aggregate = vocab_data.get('is_aggregate', False)
        self.lang_offsets = vocab_data.get('lang_offsets', {})

        self.sp = None
        self.sp_tokenizers: Dict = {}

        if tokenizer_dir:
            self._load_tokenizers(Path(tokenizer_dir))

    def _load_tokenizers(self, tokenizer_dir: Path) -> None:
        try:
            import sentencepiece as spm

            if self.is_aggregate:
                for lang, offset_info in self.lang_offsets.items():
                    model_file = tokenizer_dir / f'tokenizer_{lang}.model'
                    if model_file.exists():
                        sp = spm.SentencePieceProcessor()
                        sp.Load(str(model_file))
                        self.sp_tokenizers[lang] = {
                            'tokenizer': sp,
                            'offset': offset_info['offset'],
                            'size': offset_info['size']
                        }
            else:
                model_file = tokenizer_dir / 'tokenizer.model'
                if model_file.exists():
                    self.sp = spm.SentencePieceProcessor()
                    self.sp.Load(str(model_file))
                    self.tokenizer_type = 'bpe'

        except ImportError:
            print("Warning: sentencepiece not installed. Using vocabulary-based decoding.")

    def _decode_aggregate(self, decoded_ids: List[int]) -> str:
        if not self.sp_tokenizers:
            return self._decode_vocabulary(decoded_ids)

        segments = []
        current_lang = None
        current_ids = []

        for token_id in decoded_ids:
            lang = None
            local_id = token_id

            for lang_name, info in self.lang_offsets.items():
                if info['offset'] <= token_id < info['offset'] + info['size']:
                    lang = lang_name
                    local_id = token_id - info['offset']
                    break

            if lang is None:
                continue

            if lang == current_lang:
                current_ids.append(local_id)
            else:
                if current_ids and current_lang in self.sp_tokenizers:
                    sp = self.sp_tokenizers[current_lang]['tokenizer']
                    segments.append(sp.DecodeIds(current_ids))
                current_lang = lang
                current_ids = [local_id]

        if current_ids and current_lang in self.sp_tokenizers:
            sp = self.sp_tokenizers[current_lang]['tokenizer']
            segments.append(sp.DecodeIds(current_ids))

        return ' '.join(segments).strip()

    def _decode_vocabulary(self, decoded_ids: List[int]) -> str:
        chars = []
        for idx in decoded_ids:
            if 0 <= idx < len(self.vocabulary):
                chars.append(self.vocabulary[idx])
        text = ''.join(chars)
        text = text.replace('▁', ' ')
        text = text.replace('|', ' ')
        return text.strip()

    def _ctc_collapse(self, logits: np.ndarray):
        predictions = np.argmax(logits, axis=-1)

        decoded_ids = []
        offsets = []
        prev_id = -1
        run_start = 0

        for i, pred_id in enumerate(predictions):
            if pred_id != prev_id:
                if prev_id != self.blank_id and prev_id != -1:
                    offsets.append((int(prev_id), run_start, i - 1))
                    decoded_ids.append(int(prev_id))
                run_start = i
            prev_id = pred_id

        if prev_id != self.blank_id and prev_id != -1:
            offsets.append((int(prev_id), run_start, len(predictions) - 1))
            decoded_ids.append(int(prev_id))

        return decoded_ids, offsets

    def _ids_to_text(self, decoded_ids):
        if self.is_aggregate and self.sp_tokenizers:
            return self._decode_aggregate(decoded_ids)
        if self.sp is not None:
            return self.sp.DecodeIds(decoded_ids)
        return self._decode_vocabulary(decoded_ids)

    def decode(self, logits: np.ndarray) -> str:
        decoded_ids, _ = self._ctc_collapse(logits)
        return self._ids_to_text(decoded_ids)

    def decode_with_offsets(self, logits: np.ndarray):
        decoded_ids, offsets = self._ctc_collapse(logits)
        text = self._ids_to_text(decoded_ids)
        return text, offsets


_META_PREFIXES = ("AGE_", "GENDER_", "EMOTION_", "INTENT_", "ENTITY_", "DIALECT_", "INTENT-", "ENTITY-")
_SPECIAL_TOKENS = {
    "<unk>", "<s>", "</s>", "END",
    "NEUTRAL", "SAD", "DISGUST", "HAPPY", "FEAR", "SURPRISE", "ANGRY",
}


def _is_metadata_token(token: str) -> bool:
    return token in _SPECIAL_TOKENS or any(token.startswith(p) for p in _META_PREFIXES)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    max_val = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_val
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - log_sum_exp


class CTCBeamSearchDecoder:
    """CTC beam-search decoder backed by pyctcdecode + optional KenLM LM."""

    def __init__(
        self,
        vocab_path: str,
        tokenizer_dir: Optional[str] = None,
        blank_id: Optional[int] = None,
        kenlm_model_path: Optional[str] = None,
        unigrams_path: Optional[str] = None,
        beam_width: int = 100,
        alpha: float = 0.5,
        beta: float = 1.0,
        word_boundary_bias: float = 5.0,
    ):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        vocabulary: List[str] = vocab_data.get("vocabulary", vocab_data.get("labels", []))
        self.blank_id: int = (
            blank_id if blank_id is not None
            else vocab_data.get("blank_id", len(vocabulary))
        )
        self.beam_width = beam_width

        labels: List[str] = list(vocabulary)
        while len(labels) <= self.blank_id:
            labels.append("")
        labels[self.blank_id] = ""
        self._labels = labels

        self._meta_mask: np.ndarray = np.zeros(len(labels), dtype=bool)
        for i, tok in enumerate(labels):
            if i != self.blank_id and _is_metadata_token(tok):
                self._meta_mask[i] = True

        self._wb_bias = word_boundary_bias
        self._wb_bias_mask: Optional[np.ndarray] = None
        if word_boundary_bias > 0:
            label_set = set(labels)
            self._wb_bias_mask = np.zeros(len(labels), dtype=np.float32)
            for i, tok in enumerate(labels):
                if tok.startswith("▁") and len(tok) > 1:
                    base = tok[1:]
                    if base in label_set:
                        self._wb_bias_mask[i] = word_boundary_bias
            if not self._wb_bias_mask.any():
                self._wb_bias_mask = None

        unigrams: Optional[List[str]] = None
        if unigrams_path and Path(unigrams_path).exists():
            with open(unigrams_path, "r", encoding="utf-8") as f:
                unigrams = [line.strip() for line in f if line.strip()]
            print(f"Beam search: loaded {len(unigrams)} unigrams from {unigrams_path}")

        self._default_alpha = alpha
        self._default_beta = beta

        from pyctcdecode import build_ctcdecoder
        self._decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=kenlm_model_path,
            unigrams=unigrams,
            alpha=alpha,
            beta=beta,
        )
        lm_status = f"KenLM={kenlm_model_path}" if kenlm_model_path else "no LM"
        print(f"Beam search decoder ready (beam={beam_width}, alpha={alpha}, beta={beta}, {lm_status})")

    def _prepare_logits(self, logits: np.ndarray) -> np.ndarray:
        log_probs = _log_softmax(logits)

        if log_probs.shape[-1] >= len(self._meta_mask):
            log_probs[:, self._meta_mask] = -1e4
        else:
            mask = self._meta_mask[: log_probs.shape[-1]]
            log_probs[:, mask] = -1e4

        if self._wb_bias_mask is not None:
            bias_len = min(log_probs.shape[-1], len(self._wb_bias_mask))
            log_probs[:, :bias_len] += self._wb_bias_mask[:bias_len]

        if log_probs.shape[-1] < len(self._meta_mask):
            pad_width = len(self._meta_mask) - log_probs.shape[-1]
            padding = np.full(
                (log_probs.shape[0], pad_width), -1e4, dtype=log_probs.dtype
            )
            log_probs = np.concatenate([log_probs, padding], axis=-1)

        return log_probs

    def decode(
        self,
        logits: np.ndarray,
        hotwords: Optional[List[str]] = None,
        hotword_weight: float = 10.0,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> str:
        if alpha is not None or beta is not None:
            self._decoder.reset_params(alpha=alpha, beta=beta)

        logits = self._prepare_logits(logits)
        kwargs: Dict = {"beam_width": self.beam_width}
        if hotwords:
            kwargs["hotwords"] = hotwords
            kwargs["hotword_weight"] = hotword_weight
        try:
            text = self._decoder.decode(logits, **kwargs)
        except ValueError:
            predictions = np.argmax(logits, axis=-1)
            tokens = []
            prev = -1
            blank_id = logits.shape[-1] - 1
            for p in predictions:
                if p != prev and p != blank_id:
                    if 0 <= p < len(self._labels):
                        tokens.append(self._labels[p])
                prev = p
            text = "".join(tokens)

        if alpha is not None or beta is not None:
            self._decoder.reset_params(alpha=self._default_alpha, beta=self._default_beta)

        text = text.replace("▁", " ").replace("|", " ")
        return " ".join(text.split()).strip()

    def decode_batch(
        self,
        logits_list: List[np.ndarray],
        hotwords: Optional[List[str]] = None,
        hotword_weight: float = 10.0,
    ) -> List[str]:
        processed = [self._prepare_logits(lg) for lg in logits_list]
        kwargs: Dict = {"beam_width": self.beam_width}
        if hotwords:
            kwargs["hotwords"] = hotwords
            kwargs["hotword_weight"] = hotword_weight
        try:
            with self._decoder.pool():
                results = self._decoder.decode_batch(processed, **kwargs)
        except ValueError:
            results = [self.decode(lg) for lg in logits_list]
            return results
        return [r.replace("▁", " ").strip() for r in results]
