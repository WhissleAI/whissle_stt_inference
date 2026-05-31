#!/usr/bin/env bash
set -eo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$DIR/.venv"

# ── Model Registry ─────────────────────────────────────────────────
get_hf_repo() {
    case "$1" in
        en-meta)        echo "WhissleAI/STT-meta-1B" ;;
        zh)             echo "WhissleAI/STT-zh-mandarin-ONNX" ;;
        hinglish-loans) echo "WhissleAI/STT-hinglish-loans-ONNX" ;;
        en-in-tech-misc) echo "WhissleAI/STT-en-in-tech-misc-ONNX" ;;
        gj)             echo "WhissleAI/STT-gujlish-ONNX" ;;
        slurp)          echo "WhissleAI/STT-slurp-intent-ONNX" ;;
        *)              echo "" ;;
    esac
}

get_desc() {
    case "$1" in
        en-meta)        echo "Multilingual (9 langs) · 512d Conformer · dual-head · 488 MB" ;;
        zh)             echo "Mandarin Chinese · 1024d Conformer · dual-head · 600 MB" ;;
        hinglish-loans) echo "Hindi-English code-mixed · 512d Conformer · dual-head · 478 MB" ;;
        en-in-tech-misc) echo "English-Indian · 512d Conformer · dual-head (6 categories) · 484 MB" ;;
        gj)             echo "Gujarati-English · wav2vec2 · transcription only · 363 MB" ;;
        slurp)          echo "English SLURP · 512d Conformer · inline intent tokens · 496 MB" ;;
        *)              echo "" ;;
    esac
}

ALL_MODELS="en-meta zh hinglish-loans en-in-tech-misc gj slurp"

# ── Usage ───────────────────────────────────────────────────────────
usage() {
    echo "Whissle STT Inference — Setup & Run"
    echo ""
    echo "Usage: ./setup.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model MODEL    Model to download and run (default: en-meta)"
    echo "  --list           List all available models"
    echo "  --download-only  Download model without starting the server"
    echo "  --all            Download all models"
    echo "  --port PORT      Server port (default: 8001)"
    echo "  --token TOKEN    HuggingFace token (or set HF_TOKEN env var)"
    echo "  --help           Show this help"
    echo ""
    echo "Models are downloaded from HuggingFace (gated — license acceptance required)."
    echo ""
    echo "First time setup:"
    echo "  1. Accept the license: https://huggingface.co/WhissleAI/STT-meta-1B"
    echo "  2. Get a token: https://huggingface.co/settings/tokens"
    echo "  3. Run: ./setup.sh --token hf_your_token_here"
    echo ""
    echo "Examples:"
    echo "  ./setup.sh --token hf_xxx                         # en-meta (default)"
    echo "  ./setup.sh --model hinglish-loans --token hf_xxx  # Hinglish model"
    echo "  ./setup.sh --model zh --port 8002                 # Mandarin on port 8002"
    echo "  ./setup.sh --all --download-only                  # Download all models"
    echo ""
}

list_models() {
    echo ""
    echo "Available models:"
    echo ""
    for mid in $ALL_MODELS; do
        local status=""
        if [ -f "$DIR/models/$mid/model.onnx" ]; then
            status=" [downloaded]"
        fi
        printf "  %-20s %s%s\n" "$mid" "$(get_desc $mid)" "$status"
    done
    echo ""
    echo "All models: https://huggingface.co/WhissleAI"
    echo ""
}

# ── Python environment ──────────────────────────────────────────────
setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtualenv at $VENV_DIR"
        python3 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"

    if ! python -c "import fastapi, onnxruntime, librosa, huggingface_hub" 2>/dev/null; then
        echo "Installing dependencies..."
        pip install -q -r "$DIR/requirements.txt"
    fi
}

# ── Download from HuggingFace ──────────────────────────────────────
download_model() {
    local mid="$1"
    local hf_token="$2"
    local model_dir="$DIR/models/$mid"
    local hf_repo
    hf_repo=$(get_hf_repo "$mid")

    if [ -z "$hf_repo" ]; then
        echo "Unknown model: $mid"
        list_models
        return 1
    fi

    if [ -f "$model_dir/model.onnx" ]; then
        local size
        size=$(du -sh "$model_dir" 2>/dev/null | cut -f1)
        echo "Model '$mid' already downloaded ($size) at $model_dir"
        return
    fi

    echo ""
    echo "Downloading: $mid"
    echo "  From: https://huggingface.co/$hf_repo"
    echo "  $(get_desc $mid)"
    mkdir -p "$model_dir"

    local token_arg=""
    if [ -n "$hf_token" ]; then
        token_arg="token='$hf_token',"
    fi

    "$VENV_DIR/bin/python" - "$hf_repo" "$model_dir" "$hf_token" "$mid" << 'PYEOF'
import sys, os, shutil
from huggingface_hub import snapshot_download

repo, dest, token, model_id = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

# STT-meta-1B has ONNX files in an onnx/ subfolder (repo also has .nemo)
# All other repos have ONNX files at root
has_onnx_subfolder = (model_id == "en-meta")

try:
    if has_onnx_subfolder:
        dl_path = snapshot_download(
            repo_id=repo,
            local_dir=dest + "/_hf_tmp",
            token=token or None,
            allow_patterns=["onnx/*"],
        )
        # Move files from onnx/ subfolder to dest
        onnx_dir = os.path.join(dest, "_hf_tmp", "onnx")
        if os.path.isdir(onnx_dir):
            for f in os.listdir(onnx_dir):
                src = os.path.join(onnx_dir, f)
                if os.path.isfile(src):
                    shutil.move(src, os.path.join(dest, f))
        shutil.rmtree(os.path.join(dest, "_hf_tmp"), ignore_errors=True)
    else:
        snapshot_download(
            repo_id=repo,
            local_dir=dest,
            token=token or None,
            ignore_patterns=["*.md", "LICENSE", ".gitattributes"],
        )

    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(dest) for f in fns
                if not f.startswith('.'))
    print(f"  Done: {total / 1024 / 1024:.0f} MB")
except Exception as e:
    msg = str(e)
    if '403' in msg or 'Access' in msg or 'gated' in msg.lower():
        print(f"\n  ERROR: Access denied to {repo}")
        print(f"  Accept the license first: https://huggingface.co/{repo}")
        print(f"  Then pass your HF token via --token or HF_TOKEN env var.\n")
        sys.exit(1)
    raise
PYEOF
    echo ""
}

# ── Run server ──────────────────────────────────────────────────────
run_server() {
    local mid="$1"
    local port="$2"
    local model_dir="$DIR/models/$mid"

    echo ""
    echo "Starting Whissle STT server"
    echo "  Model:     $mid"
    echo "  Demo:      http://localhost:$port"
    echo "  API docs:  http://localhost:$port/docs"
    echo ""

    ASR_MODEL_DIR="$model_dir" python -m uvicorn src.server:app \
        --host 0.0.0.0 \
        --port "$port" \
        --log-level info
}

# ── Main ────────────────────────────────────────────────────────────
main() {
    local model="${ASR_MODEL:-en-meta}"
    local port="${PORT:-8001}"
    local hf_token="${HF_TOKEN:-}"
    local download_only=false
    local download_all=false

    while [ $# -gt 0 ]; do
        case "$1" in
            --model)         model="$2"; shift 2 ;;
            --port)          port="$2"; shift 2 ;;
            --token)         hf_token="$2"; shift 2 ;;
            --list)          list_models; exit 0 ;;
            --download-only) download_only=true; shift ;;
            --all)           download_all=true; shift ;;
            --help|-h)       usage; exit 0 ;;
            *)               echo "Unknown option: $1"; usage; exit 1 ;;
        esac
    done

    # Ensure venv + deps (needed for huggingface_hub)
    setup_venv

    # Download
    if [ "$download_all" = true ]; then
        for mid in $ALL_MODELS; do
            download_model "$mid" "$hf_token"
        done
        if [ "$download_only" = true ]; then
            echo "All models downloaded."
            exit 0
        fi
        model="en-meta"
    else
        download_model "$model" "$hf_token"
        if [ "$download_only" = true ]; then
            exit 0
        fi
    fi

    run_server "$model" "$port"
}

cd "$DIR"
main "$@"
