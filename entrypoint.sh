#!/usr/bin/env bash
set -eo pipefail

MODEL_DIR="${ASR_MODEL_DIR:-/app/models}"
MODEL="${ASR_MODEL:-en-meta}"
PORT="${PORT:-8001}"

# HF repo mapping
get_hf_repo() {
    case "$1" in
        en-meta)         echo "WhissleAI/STT-meta-1B" ;;
        zh)              echo "WhissleAI/STT-zh-mandarin-ONNX" ;;
        hinglish-loans)  echo "WhissleAI/STT-hinglish-loans-ONNX" ;;
        en-in-tech-misc) echo "WhissleAI/STT-en-in-tech-misc-ONNX" ;;
        gj)              echo "WhissleAI/STT-gujlish-ONNX" ;;
        slurp)           echo "WhissleAI/STT-slurp-intent-ONNX" ;;
        *)               echo "" ;;
    esac
}

# Download model from HuggingFace if not present
if [ ! -f "$MODEL_DIR/model.onnx" ]; then
    HF_REPO=$(get_hf_repo "$MODEL")
    if [ -z "$HF_REPO" ]; then
        echo "Unknown model: $MODEL"
        exit 1
    fi

    echo "Downloading model '$MODEL' from $HF_REPO..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$HF_REPO',
    local_dir='$MODEL_DIR',
    token='${HF_TOKEN}' or None,
    ignore_patterns=['*.md', 'LICENSE', '.gitattributes'],
)
print('Download complete')
"
fi

echo "Starting Whissle STT — model: $MODEL, port: $PORT"
exec python -m uvicorn src.server:app \
    --host "${HOST:-0.0.0.0}" \
    --port "$PORT" \
    --log-level info
