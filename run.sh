#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${ASR_MODEL_DIR:-$DIR/models}"
GCS_BASE="https://storage.googleapis.com/whissle-voice-recordings/asr-models/model-en-meta"
PORT="${PORT:-8001}"
VENV_DIR="$DIR/.venv"

MODEL_FILES=(
    model.onnx
    model.onnx.data
    tag_classifier.onnx
    tag_classifier.onnx.data
    tag_classifier.json
    config.json
    vocabulary.json
    tokenizer.model
    silero_vad.onnx
)

# ── Models ──────────────────────────────────────────────────────────
download_models() {
    if [ -f "$MODEL_DIR/model.onnx" ] && [ -f "$MODEL_DIR/model.onnx.data" ]; then
        echo "Models already present at $MODEL_DIR"
        return
    fi

    echo "Downloading models → $MODEL_DIR"
    mkdir -p "$MODEL_DIR"

    for f in "${MODEL_FILES[@]}"; do
        if [ -f "$MODEL_DIR/$f" ]; then
            echo "  $f (exists)"
            continue
        fi
        echo "  $f ..."
        curl -fSL --progress-bar "$GCS_BASE/$f" -o "$MODEL_DIR/$f"
    done

    echo "Download complete ($(du -sh "$MODEL_DIR" | cut -f1))"
}

# ── Python environment ──────────────────────────────────────────────
setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtualenv at $VENV_DIR"
        python3 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"

    if ! python -c "import fastapi, onnxruntime, librosa" 2>/dev/null; then
        echo "Installing dependencies..."
        pip install -q -r "$DIR/requirements.txt"
    fi
}

# ── Run ─────────────────────────────────────────────────────────────
main() {
    download_models
    setup_venv

    echo ""
    echo "Starting Whissle ASR server on http://localhost:$PORT"
    echo "Open http://localhost:$PORT in your browser for the streaming demo"
    echo ""

    ASR_MODEL_DIR="$MODEL_DIR" python -m uvicorn src.server:app \
        --host 0.0.0.0 \
        --port "$PORT" \
        --log-level info
}

cd "$DIR"
main "$@"
