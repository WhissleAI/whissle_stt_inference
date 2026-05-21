from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SILERO_CHUNK_SAMPLES_16K = 512
SILERO_CHUNK_SAMPLES_8K = 256
SILERO_SR = 16000


class SileroVAD:
    """Stateful Silero VAD wrapper over ONNX Runtime."""

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.5,
        sample_rate: int = 16000,
    ):
        import onnxruntime as ort

        self.threshold = threshold
        self.sr = sample_rate
        self.chunk_size = (
            SILERO_CHUNK_SAMPLES_16K if sample_rate >= 16000
            else SILERO_CHUNK_SAMPLES_8K
        )
        self._context_size = 64 if sample_rate >= 16000 else 32

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]

        state_input = next(
            (i for i in self._session.get_inputs() if "state" in i.name.lower()),
            None,
        )
        if state_input and state_input.shape:
            self._state_shape = tuple(
                d if isinstance(d, int) else 1 for d in state_input.shape
            )
        else:
            self._state_shape = (2, 1, 128)

        self._state: np.ndarray = np.zeros(self._state_shape, dtype=np.float32)
        self._context: np.ndarray = np.zeros(self._context_size, dtype=np.float32)

    def reset(self) -> None:
        self._state = np.zeros(self._state_shape, dtype=np.float32)
        self._context = np.zeros(self._context_size, dtype=np.float32)

    def _run_chunk(self, chunk: np.ndarray) -> float:
        x = np.concatenate([self._context, chunk])
        if x.ndim == 1:
            x = x[np.newaxis, :]
        ort_inputs = {self._input_names[0]: x.astype(np.float32)}
        for name in self._input_names[1:]:
            nl = name.lower()
            if "state" in nl:
                ort_inputs[name] = self._state
            elif "sr" in nl:
                ort_inputs[name] = np.array(self.sr, dtype=np.int64)

        outputs = self._session.run(None, ort_inputs)
        prob = float(outputs[0].flatten()[0])

        for i, oname in enumerate(self._output_names):
            if "state" in oname.lower():
                self._state = outputs[i]
                break

        self._context = chunk[-self._context_size:].copy()
        return prob

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        if audio.dtype == np.int16:
            return audio.astype(np.float32) / 32768.0
        if audio.dtype == np.float32 and len(audio) > 0 and np.abs(audio).max() > 2.0:
            return audio / 32768.0
        return audio.astype(np.float32)

    def __call__(self, audio: np.ndarray) -> float:
        audio = self._normalize(audio)
        if len(audio) < self.chunk_size:
            padded = np.zeros(self.chunk_size, dtype=np.float32)
            padded[:len(audio)] = audio
            return self._run_chunk(padded)
        max_prob = 0.0
        for start in range(0, len(audio) - self.chunk_size + 1, self.chunk_size):
            chunk = audio[start:start + self.chunk_size]
            prob = self._run_chunk(chunk)
            max_prob = max(max_prob, prob)
        return max_prob

    def speech_ratio(self, audio: np.ndarray) -> float:
        audio = self._normalize(audio)
        if len(audio) < self.chunk_size:
            prob = self._run_chunk(
                np.pad(audio, (0, self.chunk_size - len(audio)))
            )
            return 1.0 if prob >= self.threshold else 0.0
        n_chunks = 0
        active = 0
        for start in range(0, len(audio) - self.chunk_size + 1, self.chunk_size):
            chunk = audio[start:start + self.chunk_size]
            prob = self._run_chunk(chunk)
            n_chunks += 1
            if prob >= self.threshold:
                active += 1
        return active / n_chunks if n_chunks > 0 else 0.0

    def has_speech(self, audio: np.ndarray) -> bool:
        return self(audio) >= self.threshold

    def tail_is_silence(self, audio: np.ndarray, tail_samples: int) -> bool:
        if len(audio) < tail_samples:
            return False
        tail = self._normalize(audio[-tail_samples:])
        for start in range(0, len(tail) - self.chunk_size + 1, self.chunk_size):
            chunk = tail[start:start + self.chunk_size]
            prob = self._run_chunk(chunk)
            if prob >= self.threshold:
                return False
        return True


def load_silero_vad(
    model_dir: str | Path,
    threshold: float = 0.5,
    sample_rate: int = 16000,
) -> Optional[SileroVAD]:
    path = Path(model_dir) / "silero_vad.onnx"
    if not path.exists():
        logger.info("Silero VAD not found at %s — using energy fallback", path)
        return None
    try:
        vad = SileroVAD(path, threshold=threshold, sample_rate=sample_rate)
        logger.info("Silero VAD loaded from %s", path)
        return vad
    except Exception:
        logger.exception("Failed to load Silero VAD from %s", path)
        return None
