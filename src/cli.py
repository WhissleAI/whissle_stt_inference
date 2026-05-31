#!/usr/bin/env python3
"""Whissle STT command-line interface."""

import argparse
import json
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        prog="whissle-stt",
        description="Transcribe audio with metadata extraction (age, gender, emotion, intent)",
    )
    sub = parser.add_subparsers(dest="command")

    # transcribe
    t = sub.add_parser("transcribe", help="Transcribe audio file(s)")
    t.add_argument("files", nargs="+", help="Audio file(s) to transcribe")
    t.add_argument("--model-dir", help="Path to model directory")
    t.add_argument("--model", default="en-meta", help="Model ID (default: en-meta)")
    t.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    t.add_argument("--language", default="", help="Language code hint")
    t.add_argument("--lm", action="store_true", help="Enable language model")
    t.add_argument("--json", action="store_true", dest="json_output", help="Output raw JSON")

    # serve
    s = sub.add_parser("serve", help="Start the inference server")
    s.add_argument("--model-dir", help="Path to model directory")
    s.add_argument("--model", default="en-meta", help="Model ID")
    s.add_argument("--port", type=int, default=8001)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--device", default="cpu")

    # models
    sub.add_parser("models", help="List available models")

    args = parser.parse_args()

    if args.command == "transcribe":
        from src import transcribe
        for fpath in args.files:
            t0 = time.time()
            result = transcribe(
                fpath,
                model_dir=args.model_dir,
                model=args.model,
                device=args.device,
                language=args.language,
                use_lm=args.lm,
            )
            elapsed = time.time() - t0

            if args.json_output:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"\n{fpath}")
                print(f"  {result.get('transcript', '')}")
                tags = result.get("tags", {})
                if tags:
                    tag_str = " | ".join(f"{k}: {v}" for k, v in sorted(tags.items()) if v)
                    print(f"  [{tag_str}]")
                dur = result.get("duration", 0)
                print(f"  {dur:.1f}s audio, {elapsed:.2f}s inference")

    elif args.command == "serve":
        import os
        os.environ.setdefault("ASR_MODEL_DIR", args.model_dir or f"models/{args.model}")
        os.environ.setdefault("PORT", str(args.port))
        os.environ.setdefault("ASR_DEVICE", args.device)
        import uvicorn
        uvicorn.run("src.server:app", host=args.host, port=args.port, log_level="info")

    elif args.command == "models":
        models = {
            "en-meta": "Multilingual (9 langs) · dual-head · 488 MB",
            "zh": "Mandarin Chinese · dual-head · 600 MB",
            "hinglish-loans": "Hindi-English code-mixed · dual-head · 478 MB",
            "en-in-tech-misc": "English-Indian · dual-head (6 categories) · 484 MB",
            "gj": "Gujarati-English · wav2vec2 · 363 MB",
            "slurp": "English SLURP · inline intents · 496 MB",
        }
        print("\nAvailable models:\n")
        for mid, desc in models.items():
            print(f"  {mid:<20} {desc}")
        print(f"\nDownload: ./setup.sh --model <id> --token <hf_token>")
        print(f"HuggingFace: https://huggingface.co/WhissleAI\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
