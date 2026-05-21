from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .vad import SileroVAD

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MIN_CHUNK_SEC = 0.8
MAX_CHUNK_SEC = 8.0
MIN_CHUNK_SAMPLES = int(MIN_CHUNK_SEC * SAMPLE_RATE)
MAX_CHUNK_SAMPLES = int(MAX_CHUNK_SEC * SAMPLE_RATE)
MIN_SAMPLES_TO_TRANSCRIBE = int(0.3 * SAMPLE_RATE)
VALID_SAMPLE_RATES = {8000, 16000, 22050, 44100, 48000}
ENERGY_VAD_THRESHOLD = 200
SILENCE_TAIL_SAMPLES = int(0.4 * SAMPLE_RATE)
LOOKBACK_SAMPLES = int(0.5 * SAMPLE_RATE)
MIN_VOICE_RATIO = 0.05

INTERIM_INTERVAL_SAMPLES = int(0.25 * SAMPLE_RATE)
INTERIM_MIN_SAMPLES = int(0.25 * SAMPLE_RATE)

MAX_SENTENCE_WORDS = 50
MAX_SENTENCE_SEC = 6.0


_ASR_ARTIFACT_RE = re.compile(
    r"SPEAKER\s*(?:<unk>\s*)?CHANGE"
    r"|SPEAK\s*(?:<unk>\s*)?CHANGE"
    r"|SPER\s*(?:<unk>\s*)?(?:CHANGE|NGE|CHA)"
    r"|EAKER\s*(?:<unk>\s*)?(?:CHANGE|NGE|CHA)"
    r"|<unk>(?:CHANGE|NGE|CHA)?"
    r"|<unk>\s*"
    r"|ENTITY_\w+\s*"
    r"|EMOTION_\w+\s*"
    r"|INTENT_\w+\s*"
    r"|AGE_[<>]?\w+\s*"
    r"|GENDER_\w+\s*"
    r"|DIALECT_\w+\s*"
    r"|\bEND\b",
)

_GLUED_ARTIFACT_RE = re.compile(
    r"EMOTION_\w+|INTENT_\w+|ENTITY_\w+|AGE_[<>]?\w+|GENDER_\w+|DIALECT_\w+",
)

_SPEAKER_CHANGE_RE = re.compile(
    r"SPEAKER\s*(?:<unk>\s*)?CHANGE"
    r"|SPEAK\s*(?:<unk>\s*)?CHANGE"
    r"|SPER\s*(?:<unk>\s*)?(?:CHANGE|NGE|CHA)"
    r"|EAKER\s*(?:<unk>\s*)?(?:CHANGE|NGE|CHA)",
)


def clean_asr_text(raw: str) -> str:
    text = _ASR_ARTIFACT_RE.sub(" ", raw)
    text = _GLUED_ARTIFACT_RE.sub("", text)
    text = text.replace('|', ' ')
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def split_by_speaker_change(raw: str) -> List[str]:
    parts = _SPEAKER_CHANGE_RE.split(raw)
    return [s for s in (clean_asr_text(p) for p in parts) if s]


def _is_transcript_empty(text: str) -> bool:
    t = text.strip()
    return not t or t == "."


_LEGIT_SHORT_WORDS = frozenset({
    "yes", "no", "yeah", "yep", "nah", "nope", "ok", "okay", "hey", "hi",
    "bye", "sure", "right", "wow", "nice", "good", "bad", "fine", "cool",
    "great", "stop", "go", "wait", "what", "why", "how", "who", "when",
    "where", "done", "next", "help", "please", "thanks", "sorry", "hello",
})


def _is_garbage_text(text: str) -> bool:
    words = text.split()
    if not words:
        return True
    stripped = text.strip()
    if not stripped or len(stripped) < 2:
        return True
    if len(words) == 1:
        w = words[0].lower().rstrip(".,!?")
        return len(w) < 3 and w not in _LEGIT_SHORT_WORDS
    if all(len(w) <= 2 for w in words):
        return True
    return False


@dataclass
class TranscriptSegment:
    channel: str
    text: str
    audio_offset: float
    is_final: bool = True
    utterance_end: bool = False
    metadata: Optional[Dict[str, str]] = None
    metadata_probs: Optional[Dict[str, Any]] = None
    entities: Optional[List[Dict[str, str]]] = None
    speaker_change: bool = False
    tags: Optional[Dict[str, str]] = None
    process_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "type": "transcript",
            "channel": self.channel,
            "text": self.text,
            "audioOffset": round(self.audio_offset, 3),
            "is_final": self.is_final,
            "utterance_end": self.utterance_end,
        }
        if self.process_ms is not None:
            d["process_ms"] = round(self.process_ms)
        if self.metadata:
            d["metadata"] = self.metadata
        if self.metadata_probs:
            d["metadata_probs"] = self.metadata_probs
        if self.entities:
            d["entities"] = self.entities
        if self.speaker_change:
            d["speakerChange"] = True
        if self.tags:
            d["tags"] = self.tags
        return d


@dataclass
class StreamingConfig:
    language: str = ""
    use_lm: bool = True
    sample_rate: int = SAMPLE_RATE
    metadata_prob: bool = True
    top_k: int = 5
    beam_width: Optional[int] = None
    lm_alpha: Optional[float] = None
    lm_beta: Optional[float] = None
    hotwords: List[str] = field(default_factory=list)
    hotword_weight: float = 10.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StreamingConfig":
        raw_sr = int(d.get("sample_rate", SAMPLE_RATE))
        if raw_sr not in VALID_SAMPLE_RATES:
            logger.warning("Invalid sample_rate %d, defaulting to %d", raw_sr, SAMPLE_RATE)
            raw_sr = SAMPLE_RATE

        raw_hotwords = d.get("hotwords", [])
        hotwords: List[str] = []
        if isinstance(raw_hotwords, list):
            hotwords = [str(w) for w in raw_hotwords[:200] if w]

        raw_alpha = d.get("lm_alpha")
        lm_alpha = float(raw_alpha) if raw_alpha is not None else None
        raw_beta = d.get("lm_beta")
        lm_beta = float(raw_beta) if raw_beta is not None else None

        return cls(
            language=str(d.get("language", "")),
            use_lm=bool(d.get("use_lm", True)),
            sample_rate=raw_sr,
            metadata_prob=bool(d.get("metadata_prob", True)),
            top_k=int(d.get("top_k", 5)),
            beam_width=d.get("beam_width"),
            lm_alpha=lm_alpha,
            lm_beta=lm_beta,
            hotwords=hotwords,
            hotword_weight=float(d.get("hotword_weight", 10.0)),
        )


class StreamingSession:
    """
    Accumulates raw PCM audio, detects chunk boundaries via VAD + silence tail,
    and transcribes each chunk via the ASR engine.
    """

    def __init__(
        self,
        engine: Any,
        config: StreamingConfig | None = None,
        silero_vad: SileroVAD | None = None,
    ):
        self.engine = engine
        self.config = config or StreamingConfig()
        self.channel: str = "microphone"
        self._silero_vad = silero_vad

        sr = self.config.sample_rate or SAMPLE_RATE
        self._sr = sr

        self._min_chunk_samples = int(MIN_CHUNK_SEC * sr)
        self._max_chunk_samples = int(MAX_CHUNK_SEC * sr)
        self._silence_tail_samples = int(SILENCE_TAIL_SAMPLES * sr / SAMPLE_RATE)
        self._lookback_samples = int(LOOKBACK_SAMPLES * sr / SAMPLE_RATE)
        self._interim_interval_samples = int(INTERIM_INTERVAL_SAMPLES * sr / SAMPLE_RATE)
        self._interim_min_samples = int(INTERIM_MIN_SAMPLES * sr / SAMPLE_RATE)
        self._min_samples_to_transcribe = int(MIN_SAMPLES_TO_TRANSCRIBE * sr / SAMPLE_RATE)

        self._buffers: Dict[str, np.ndarray] = {}
        self._total_samples: Dict[str, int] = {}
        self._last_interim_len: Dict[str, int] = {}
        self._voice_started: Dict[str, bool] = {}

        self._acc_text: Dict[str, List[str]] = {}
        self._acc_offset: Dict[str, float] = {}
        self._acc_meta: Dict[str, Optional[Dict[str, str]]] = {}
        self._acc_probs: Dict[str, Optional[Dict[str, Any]]] = {}
        self._acc_entities: Dict[str, Optional[List[Dict[str, str]]]] = {}
        self._acc_tags: Dict[str, Optional[Dict[str, str]]] = {}
        self._acc_process_ms: Dict[str, float] = {}

    def _get_buffer(self, ch: str) -> np.ndarray:
        if ch not in self._buffers:
            self._buffers[ch] = np.empty(0, dtype=np.float32)
            self._total_samples[ch] = 0
        return self._buffers[ch]

    def set_channel(self, name: str) -> None:
        self.channel = name

    def feed(self, pcm_bytes: bytes) -> List[TranscriptSegment]:
        if not pcm_bytes:
            return []
        usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
        if usable == 0:
            return []
        if usable < len(pcm_bytes):
            pcm_bytes = pcm_bytes[:usable]

        ch = self.channel
        buf = self._get_buffer(ch)
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        buf = np.concatenate([buf, samples])
        self._buffers[ch] = buf
        self._total_samples[ch] = self._total_samples.get(ch, 0) + len(samples)

        if not self._voice_started.get(ch, False):
            if self._has_voice_activity(buf):
                self._voice_started[ch] = True
            elif len(buf) > self._lookback_samples:
                if len(samples) > self._sr and self._has_voice_anywhere(buf):
                    self._voice_started[ch] = True
                else:
                    trimmed = buf[-self._lookback_samples:]
                    self._buffers[ch] = trimmed
                    buf = trimmed

        segments: List[TranscriptSegment] = []
        while self._should_flush(buf, ch):
            continuous_speech = len(buf) >= self._max_chunk_samples
            chunk_segs = self._transcribe_chunk(ch)
            for seg in chunk_segs:
                seg.utterance_end = True
                self._accumulate(ch, seg)

            total = self._total_samples.get(ch, 0)
            acc_start = self._acc_offset.get(ch, total / self._sr)
            acc_dur = total / self._sr - acc_start

            should_emit = (
                not continuous_speech
                or self._sentence_word_count(ch) >= MAX_SENTENCE_WORDS
                or acc_dur >= MAX_SENTENCE_SEC
            )
            if should_emit:
                segments.extend(self._emit_sentence(ch))

            buf = self._get_buffer(ch)
        return segments

    def flush(self, force: bool = True) -> List[TranscriptSegment]:
        out: List[TranscriptSegment] = []
        all_channels = set(self._buffers.keys()) | set(self._acc_text.keys())
        for ch in list(all_channels):
            buf = self._buffers.get(ch, np.empty(0))
            if force and len(buf) >= self._min_samples_to_transcribe:
                self.channel = ch
                segs = self._transcribe_chunk(ch)
                for seg in segs:
                    seg.utterance_end = True
                    self._accumulate(ch, seg)
            out.extend(self._emit_sentence(ch))
        return out

    def should_get_interim(self) -> bool:
        ch = self.channel
        buf = self._get_buffer(ch)
        if len(buf) < self._interim_min_samples:
            return False
        new_samples = len(buf) - self._last_interim_len.get(ch, 0)
        return new_samples >= self._interim_interval_samples

    def get_interim(self) -> Optional[TranscriptSegment]:
        ch = self.channel
        buf = self._get_buffer(ch)

        if len(buf) < self._interim_min_samples:
            return None

        last = self._last_interim_len.get(ch, 0)
        new_samples = len(buf) - last
        if new_samples < self._interim_interval_samples:
            return None

        if not self._has_voice_activity(buf):
            return None
        if len(buf) > 2 * self._sr and self._voice_ratio(buf) < MIN_VOICE_RATIO:
            return None

        self._last_interim_len[ch] = len(buf)

        pcm_int16 = buf.clip(-32768, 32767).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()

        try:
            result = self.engine.transcribe_pcm(
                pcm_bytes,
                sample_rate=self.config.sample_rate,
                metadata_prob=self.config.metadata_prob,
                top_k=self.config.top_k,
                language=self.config.language or None,
                use_lm=False,
            )
        except Exception:
            logger.exception("Interim transcription failed (%d samples)", len(buf))
            return None

        raw_text = (result.get("transcript") or result.get("raw_output") or "").strip()
        if _is_transcript_empty(raw_text):
            return None

        text = clean_asr_text(raw_text)
        if _is_transcript_empty(text):
            return None

        acc_parts = self._acc_text.get(ch, [])
        if acc_parts:
            text = " ".join(acc_parts) + " " + text

        total = self._total_samples.get(ch, 0)
        offset = self._acc_offset.get(ch, max(0.0, (total - len(buf)) / self._sr))

        meta = result.get("metadata")
        m_probs = result.get("metadata_probs") if self.config.metadata_prob else None

        return TranscriptSegment(
            channel=ch,
            text=text,
            audio_offset=offset,
            is_final=False,
            metadata=meta,
            metadata_probs=m_probs,
        )

    # ------------------------------------------------------------------
    # VAD / flush
    # ------------------------------------------------------------------

    def _has_voice_anywhere(self, samples: np.ndarray) -> bool:
        if len(samples) < int(self._sr * 0.1):
            return False
        if self._silero_vad is not None:
            self._silero_vad.reset()
            return self._silero_vad.has_speech(samples)
        win = self._sr
        for start in range(0, len(samples) - win + 1, win):
            chunk = samples[start:start + win]
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            if rms > ENERGY_VAD_THRESHOLD:
                return True
        return False

    def _has_voice_activity(self, samples: np.ndarray) -> bool:
        if len(samples) < int(self._sr * 0.1):
            return False
        if self._silero_vad is not None:
            tail = samples[-min(len(samples), self._sr):]
            self._silero_vad.reset()
            return self._silero_vad.has_speech(tail)
        tail = samples[-min(len(samples), self._sr):]
        rms = float(np.sqrt(np.mean(tail.astype(np.float64) ** 2)))
        return rms > ENERGY_VAD_THRESHOLD

    def _voice_ratio(self, samples: np.ndarray, window_ms: int = 50) -> float:
        if self._silero_vad is not None:
            self._silero_vad.reset()
            return self._silero_vad.speech_ratio(samples)
        win = int(self._sr * window_ms / 1000)
        if len(samples) < win:
            return 0.0
        n_windows = len(samples) // win
        active = 0
        for i in range(n_windows):
            chunk = samples[i * win : (i + 1) * win].astype(np.float64)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms > ENERGY_VAD_THRESHOLD:
                active += 1
        return active / n_windows if n_windows > 0 else 0.0

    def _has_silence_tail(self, samples: np.ndarray) -> bool:
        if len(samples) < self._silence_tail_samples:
            return False
        if self._silero_vad is not None:
            self._silero_vad.reset()
            return self._silero_vad.tail_is_silence(samples, self._silence_tail_samples)
        tail = samples[-self._silence_tail_samples:]
        rms = float(np.sqrt(np.mean(tail.astype(np.float64) ** 2)))
        return rms < ENERGY_VAD_THRESHOLD * 0.5

    def _should_flush(self, buf: np.ndarray, ch: str) -> bool:
        n = len(buf)
        if n < self._min_chunk_samples:
            return False
        if n >= self._max_chunk_samples:
            return True
        if self._has_silence_tail(buf):
            return True
        return False

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def _transcribe_chunk(self, ch: str) -> List[TranscriptSegment]:
        t0 = time.monotonic()
        buf = self._get_buffer(ch)

        self._buffers[ch] = np.empty(0, dtype=np.float32)
        self._last_interim_len[ch] = 0
        self._voice_started[ch] = False
        total = self._total_samples.get(ch, 0)
        audio_offset = max(0.0, (total - len(buf)) / self._sr)

        pcm_int16 = buf.clip(-32768, 32767).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()

        try:
            result = self.engine.transcribe_pcm(
                pcm_bytes,
                sample_rate=self._sr,
                metadata_prob=self.config.metadata_prob,
                top_k=self.config.top_k,
                language=self.config.language or None,
                use_lm=self.config.use_lm,
                beam_width=self.config.beam_width,
                lm_alpha=self.config.lm_alpha,
                lm_beta=self.config.lm_beta,
                hotwords=self.config.hotwords or None,
                hotword_weight=self.config.hotword_weight,
            )
        except Exception:
            logger.exception("Chunk transcription failed (%d samples)", len(buf))
            return []

        raw_text = result.get("transcript", "")
        if _is_transcript_empty(raw_text):
            return []

        text = clean_asr_text(raw_text)
        if _is_transcript_empty(text) or _is_garbage_text(text):
            return []

        speaker_change = bool(_SPEAKER_CHANGE_RE.search(result.get("raw_output", "")))
        elapsed_ms = (time.monotonic() - t0) * 1000

        seg = TranscriptSegment(
            channel=ch,
            text=text,
            audio_offset=audio_offset,
            is_final=True,
            metadata=result.get("metadata"),
            metadata_probs=result.get("metadata_probs") if self.config.metadata_prob else None,
            entities=result.get("entities"),
            speaker_change=speaker_change,
            tags=result.get("tags"),
            process_ms=elapsed_ms,
        )
        return [seg]

    # ------------------------------------------------------------------
    # Sentence accumulation
    # ------------------------------------------------------------------

    def _accumulate(self, ch: str, seg: TranscriptSegment) -> None:
        if ch not in self._acc_text:
            self._acc_text[ch] = []
            total = self._total_samples.get(ch, 0)
            self._acc_offset[ch] = seg.audio_offset
            self._acc_process_ms[ch] = 0.0

        self._acc_text[ch].append(seg.text)
        self._acc_meta[ch] = seg.metadata
        self._acc_probs[ch] = seg.metadata_probs
        self._acc_entities[ch] = seg.entities
        self._acc_tags[ch] = seg.tags
        self._acc_process_ms[ch] = self._acc_process_ms.get(ch, 0.0) + (seg.process_ms or 0.0)

    def _sentence_word_count(self, ch: str) -> int:
        parts = self._acc_text.get(ch, [])
        return sum(len(p.split()) for p in parts)

    def _emit_sentence(self, ch: str) -> List[TranscriptSegment]:
        parts = self._acc_text.get(ch, [])
        if not parts:
            return []

        text = " ".join(parts).strip()
        if _is_transcript_empty(text):
            self._clear_acc(ch)
            return []

        offset = self._acc_offset.get(ch, 0.0)
        seg = TranscriptSegment(
            channel=ch,
            text=text,
            audio_offset=offset,
            is_final=True,
            utterance_end=True,
            metadata=self._acc_meta.get(ch),
            metadata_probs=self._acc_probs.get(ch),
            entities=self._acc_entities.get(ch),
            tags=self._acc_tags.get(ch),
            process_ms=self._acc_process_ms.get(ch, 0.0),
        )
        self._clear_acc(ch)
        return [seg]

    def _clear_acc(self, ch: str) -> None:
        self._acc_text.pop(ch, None)
        self._acc_offset.pop(ch, None)
        self._acc_meta.pop(ch, None)
        self._acc_probs.pop(ch, None)
        self._acc_entities.pop(ch, None)
        self._acc_tags.pop(ch, None)
        self._acc_process_ms.pop(ch, None)
