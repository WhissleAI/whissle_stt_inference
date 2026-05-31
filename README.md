# Whissle STT Inference

ONNX-based speech-to-text server with real-time streaming, metadata extraction (age, gender, emotion, intent), and multi-model support. Runs on CPU — no GPU required.

## Quick Start

```bash
git clone https://github.com/WhissleAI/whissle_stt_inference.git
cd whissle_stt_inference
./setup.sh
```

Downloads the default model (~488 MB), creates a Python virtualenv, and starts the server at http://localhost:8001.

## Available Models

| Model ID | Languages | Type | Size | Description |
|----------|-----------|------|------|-------------|
| `en-meta` | EN, HI, ES, FR, DE, IT, GU, MR | Dual-head | ~488 MB | Multilingual with meta tags (age, gender, emotion, intent) |
| `zh` | ZH (Mandarin) | Dual-head | ~600 MB | Mandarin Chinese with meta tags |
| `hinglish-loans` | HI-EN (Hinglish) | Dual-head | ~478 MB | Hindi-English code-mixed with meta tags |
| `en-in-tech-misc` | EN-IN | Dual-head | ~484 MB | English-Indian tech/misc domain with meta tags |
| `en` | EN + 22 languages | Single-head | ~2.4 GB | Large multilingual (7 tokenizer groups, KenLM support) |
| `gj` | GU-EN (Gujlish) | Single-head | ~363 MB | Gujarati-English wav2vec2 |
| `slurp` | EN | Single-head | ~496 MB | English with SLURP intent classification |

**Dual-head** models output both transcription and speaker metadata (age, gender, emotion, intent) via a tag classifier. **Single-head** models output transcription with inline metadata tokens.

## Usage

```bash
# Run with a specific model
./setup.sh --model hinglish-loans

# Run on a custom port
./setup.sh --model zh --port 8002

# Download all models without starting the server
./setup.sh --all --download-only

# List available models
./setup.sh --list
```

## Requirements

- Python 3.11+
- macOS or Linux

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Streaming demo UI |
| `GET` | `/health` | Health check |
| `GET` | `/models` | List loaded models |
| `POST` | `/asr/transcribe` | Batch transcription (file upload) |
| `WS` | `/asr/stream` | Real-time streaming WebSocket |

### Batch Transcription

```bash
curl -F "file=@recording.wav" http://localhost:8001/asr/transcribe
```

Response (dual-head model):
```json
{
  "text": "hello how are you",
  "tags": {
    "gender": "MALE",
    "age": "AGE_25_30",
    "emotion": "NEUTRAL",
    "intent": "INTENT-GREETING"
  },
  "entities": [],
  "language": "en",
  "duration": 2.5,
  "processing_time": 0.31
}
```

### WebSocket Streaming

1. Connect to `ws://localhost:8001/asr/stream`
2. Send config: `{"type": "config", "language": "en"}`
3. Send binary PCM frames (Int16LE, 16kHz, mono)
4. Receive JSON transcripts in real-time
5. Send `{"type": "end"}` to close

## Docker

```bash
# Build
docker build -t whissle-stt .

# Run with default model (en-meta)
docker run -p 8001:8001 whissle-stt

# Run with a specific model
docker run -p 8001:8001 -e ASR_MODEL=hinglish-loans whissle-stt
```

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `ASR_MODEL` | `en-meta` | Model to download and load |
| `ASR_MODEL_DIR` | `./models/<model>` | Path to model directory |
| `ASR_MODELS` | | JSON array for loading multiple models |
| `PORT` | `8001` | Server port |
| `ASR_DEVICE` | `cpu` | Inference device |
| `ASR_BEAM_WIDTH` | `100` | Beam search width |

## Multi-Model Server

Load multiple models simultaneously:

```bash
# Download models
./setup.sh --model en-meta --download-only
./setup.sh --model hinglish-loans --download-only

# Start with multiple models
ASR_MODEL_DIR=./models/en-meta \
ASR_MODEL_HINGLISH_LOANS_DIR=./models/hinglish-loans \
python -m uvicorn src.server:app --host 0.0.0.0 --port 8001
```

Select model per request: `POST /asr/transcribe?model=hinglish-loans`

## HuggingFace Models

All models are available on HuggingFace under the [Whissle Community License](https://huggingface.co/WhissleAI/STT-meta-1B/blob/main/LICENSE):

- [WhissleAI/STT-meta-1B](https://huggingface.co/WhissleAI/STT-meta-1B) — en-meta
- [WhissleAI/STT-hinglish-loans-ONNX](https://huggingface.co/WhissleAI/STT-hinglish-loans-ONNX) — hinglish-loans
- [WhissleAI/STT-zh-mandarin-ONNX](https://huggingface.co/WhissleAI/STT-zh-mandarin-ONNX) — zh
- [WhissleAI/STT-en-in-tech-misc-ONNX](https://huggingface.co/WhissleAI/STT-en-in-tech-misc-ONNX) — en-in-tech-misc
- [WhissleAI/STT-en-default-ONNX](https://huggingface.co/WhissleAI/STT-en-default-ONNX) — en
- [WhissleAI/STT-gujlish-ONNX](https://huggingface.co/WhissleAI/STT-gujlish-ONNX) — gj
- [WhissleAI/STT-slurp-intent-ONNX](https://huggingface.co/WhissleAI/STT-slurp-intent-ONNX) — slurp

## License

Models are licensed under the [Whissle Community License](https://huggingface.co/WhissleAI/STT-meta-1B/blob/main/LICENSE). Free for research and commercial use under 100M MAU.
