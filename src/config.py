import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    model_dir: str = os.environ.get("ASR_MODEL_DIR", "./models")
    models_json: str = os.environ.get("ASR_MODELS", "")
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = int(os.environ.get("PORT", "8001"))
    workers: int = int(os.environ.get("WORKERS", "1"))
    max_concurrent: int = int(os.environ.get("MAX_CONCURRENT", str(os.cpu_count() or 4)))
    device: str = os.environ.get("ASR_DEVICE", "cpu")
    default_language: str = os.environ.get("ASR_DEFAULT_LANG", "en")
    beam_width: int = int(os.environ.get("ASR_BEAM_WIDTH", "100"))
    lm_alpha: float = float(os.environ.get("ASR_LM_ALPHA", "0.1"))
    lm_beta: float = float(os.environ.get("ASR_LM_BETA", "0.5"))
    silence_padding: float = float(os.environ.get("ASR_SILENCE_PAD", "0.3"))


settings = Settings()
