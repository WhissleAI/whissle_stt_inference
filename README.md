# Whissle ASR Inference Server

ONNX-based ASR server with real-time streaming transcription and metadata extraction (age, gender, emotion, intent).

## Quick Start

```bash
git clone https://github.com/WhissleAI/asr_onnx_inference.git
cd asr_onnx_inference
./run.sh
```

Open http://localhost:8001 in your browser.

The script downloads the model (~488 MB) on first run, creates a Python virtualenv, installs dependencies, and starts the server. Subsequent runs skip the download.

## Requirements

- Python 3.11+
- macOS or Linux

## What It Does

- Transcribes speech in real-time via WebSocket streaming
- Extracts speaker metadata per utterance: **age**, **gender**, **emotion**, **intent**
- Uses Silero VAD for automatic utterance boundary detection
- Supports batch transcription via REST API

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Streaming demo UI |
| `GET` | `/health` | Health check |
| `GET` | `/models` | List loaded models |
| `POST` | `/asr/transcribe` | Batch transcription (file upload) |
| `WS` | `/asr/stream` | Streaming WebSocket |

### Batch Example

```bash
curl -F "file=@recording.wav" http://localhost:8001/asr/transcribe
```

### WebSocket Protocol

1. Connect to `ws://localhost:8001/asr/stream`
2. Send config: `{"type": "config", "language": "en", "use_lm": true}`
3. Send binary PCM frames (Int16LE, 16kHz, mono)
4. Receive JSON: `{"type": "transcript", "text": "...", "tags": {"gender": "MALE", "emotion": "NEUTRAL", ...}, "is_final": true}`
5. Send `{"type": "end"}` to close

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `PORT` | `8001` | Server port |
| `ASR_MODEL_DIR` | `./models` | Path to model directory |

## Model

en-meta-v1: Conformer-CTC Large (512d, 18 layers) fine-tuned with dual-head architecture — CTC decoder for transcription + tag classifier for speaker metadata. 128-token BPE vocabulary.
