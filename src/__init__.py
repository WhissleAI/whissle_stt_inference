"""Whissle STT — ONNX-based speech-to-text with metadata extraction."""

from .engine import ASREngine

__version__ = "0.1.0"
__all__ = ["ASREngine", "transcribe", "load_model"]

_default_engine = None


def load_model(model_dir: str, device: str = "cpu", **kwargs) -> ASREngine:
    """Load an ASR model from a directory.

    Args:
        model_dir: Path to directory containing model.onnx + config.json
        device: "cpu", "cuda", or "auto"

    Returns:
        ASREngine instance ready for transcription
    """
    return ASREngine(model_dir=model_dir, device=device, **kwargs)


def transcribe(
    audio,
    model_dir: str = None,
    model: str = "en-meta",
    device: str = "cpu",
    language: str = "",
    use_lm: bool = False,
    **kwargs,
) -> dict:
    """Transcribe audio with metadata extraction.

    Args:
        audio: file path (str), bytes, or numpy array (float32, 16kHz mono)
        model_dir: explicit path to model directory (overrides model param)
        model: model ID to auto-download from HuggingFace if not present
        device: "cpu", "cuda", or "auto"
        language: language code hint (optional)
        use_lm: enable language model decoding

    Returns:
        dict with keys: transcript, tags, metadata, entities, duration, processing_time

    Example:
        >>> from src import transcribe
        >>> result = transcribe("audio.wav", model="en-meta")
        >>> print(result["transcript"])
        >>> print(result["tags"])  # {"age": "30_45", "gender": "FEMALE", ...}
    """
    global _default_engine

    if model_dir is None:
        import os
        base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", model)
        if os.path.isdir(base) and os.path.isfile(os.path.join(base, "model.onnx")):
            model_dir = base
        else:
            raise FileNotFoundError(
                f"Model '{model}' not found at {base}. "
                f"Download it first: ./setup.sh --model {model} --token YOUR_HF_TOKEN --download-only"
            )

    if _default_engine is None or str(_default_engine.model_dir) != model_dir:
        _default_engine = ASREngine(model_dir=model_dir, device=device)

    if isinstance(audio, str):
        with open(audio, "rb") as f:
            audio_bytes = f.read()
    elif isinstance(audio, bytes):
        audio_bytes = audio
    else:
        import numpy as np
        if isinstance(audio, np.ndarray):
            return _default_engine.transcribe(
                audio, sample_rate=16000, language=language or None,
                use_lm=use_lm, **kwargs,
            )
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    return _default_engine.transcribe(
        audio_bytes, language=language or None, use_lm=use_lm, **kwargs,
    )
