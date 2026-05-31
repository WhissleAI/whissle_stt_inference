#!/usr/bin/env python3
"""Upload ONNX models to HuggingFace with Whissle Community License gating.

Downloads model files from GCS and uploads to a gated HF repo.

Usage:
    python scripts/upload_hf.py --model hinglish-loans --hf-token $HF_TOKEN
    python scripts/upload_hf.py --model all --hf-token $HF_TOKEN
    python scripts/upload_hf.py --list
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    print("pip install huggingface_hub")
    sys.exit(1)


MODELS = {
    "hinglish-loans": {
        "hf_repo": "WhissleAI/STT-hinglish-loans-ONNX",
        "gcs_dir": "model-hinglish-loans",
        "name": "Whissle STT Hinglish-Loans",
        "languages": ["hi", "en"],
        "language_names": "Hindi-English (Hinglish code-mixed)",
        "description": "Hindi-English code-mixed (Hinglish) ASR with meta tags. Dual-head model with tag classifier for age, gender, emotion, and intent detection.",
        "head_type": "Dual-head",
        "base_model": "nvidia/parakeet-ctc-0.6b",
        "params": "600M",
        "files": [
            "model.onnx", "config.json", "vocabulary.json",
            "tokenizer_INDO_ARYAN.model",
            "tag_classifier.onnx", "tag_classifier.json",
        ],
    },
    "zh": {
        "hf_repo": "WhissleAI/STT-zh-mandarin-ONNX",
        "gcs_dir": "model-zh",
        "name": "Whissle STT Mandarin Chinese",
        "languages": ["zh"],
        "language_names": "Mandarin Chinese",
        "description": "Mandarin Chinese ASR with meta tags. Dual-head model with tag classifier for age, gender, emotion, and intent detection.",
        "head_type": "Dual-head",
        "base_model": "nvidia/parakeet-ctc-0.6b",
        "params": "600M",
        "files": [
            "model.onnx", "config.json", "vocabulary.json",
            "tokenizer.model",
            "tag_classifier.onnx", "tag_classifier.json",
        ],
    },
    "en-in-tech-misc": {
        "hf_repo": "WhissleAI/STT-en-in-tech-misc-ONNX",
        "gcs_dir": "model-en-in-tech-misc",
        "name": "Whissle STT English-Indian Tech",
        "languages": ["en"],
        "language_names": "English (Indian accent, tech/misc domain)",
        "description": "English-Indian accent ASR optimized for tech and miscellaneous domains. Dual-head model with tag classifier for age, gender, emotion, and intent detection.",
        "head_type": "Dual-head",
        "base_model": "nvidia/parakeet-ctc-0.6b",
        "params": "600M",
        "files": [
            "model.onnx", "config.json", "vocabulary.json",
            "tokenizer_ENGLISH.model",
            "tag_classifier.onnx", "tag_classifier.json",
        ],
    },
    "en": {
        "hf_repo": "WhissleAI/STT-en-default-ONNX",
        "gcs_dir": "model-en",
        "name": "Whissle STT Multilingual Default",
        "languages": ["en", "hi", "es", "fr", "de", "it", "zh", "gu", "mr", "pa", "bn", "ta", "te", "kn", "ml", "ru", "pl", "cs", "uk", "pt", "nl", "ja", "ko"],
        "language_names": "English + 22 languages (multilingual)",
        "description": "Large multilingual ASR with 7 tokenizer groups covering 23 languages. Single-head model with inline metadata tokens and intent group classification.",
        "head_type": "Single-head",
        "base_model": "nvidia/parakeet-ctc-0.6b",
        "params": "1B",
        "files": [
            "model.onnx", "model.onnx.data", "config.json", "vocabulary.json",
            "intent_groups.json", "silero_vad.onnx",
            "tokenizer_ENGLISH.model", "tokenizer_EUROPEAN.model",
            "tokenizer_INDO_ARYAN.model", "tokenizer_DRAVIDIAN.model",
            "tokenizer_SLAVIC.model", "tokenizer_MANDARIN.model",
            "tokenizer_INDIAN_LOW_RESOURCE.model",
        ],
    },
    "gj": {
        "hf_repo": "WhissleAI/STT-gujlish-ONNX",
        "gcs_dir": "model-gj",
        "name": "Whissle STT Gujlish",
        "languages": ["gu", "en"],
        "language_names": "Gujarati-English (Gujlish)",
        "description": "Gujarati-English (Gujlish) ASR based on wav2vec2 architecture. Single-head model for code-mixed Gujarati and English transcription.",
        "head_type": "Single-head",
        "base_model": "facebook/wav2vec2-large",
        "params": "300M",
        "files": [
            "model.onnx", "config.json", "vocabulary.json",
        ],
    },
    "slurp": {
        "hf_repo": "WhissleAI/STT-slurp-intent-ONNX",
        "gcs_dir": "model-slurp",
        "name": "Whissle STT SLURP Intent",
        "languages": ["en"],
        "language_names": "English",
        "description": "English ASR with SLURP intent classification. Single-head model trained on the SLURP dataset for intent detection in spoken language understanding.",
        "head_type": "Single-head",
        "base_model": "nvidia/parakeet-ctc-0.6b",
        "params": "600M",
        "files": [
            "model.onnx", "model.onnx.data", "config.json", "vocabulary.json",
            "tokenizer.model", "intent_groups.json",
        ],
    },
}

GCS_BUCKET = "gs://whissle-voice-recordings/asr-models"

WHISSLE_LICENSE = """WHISSLE INFERENCE-ONLY LICENSE AGREEMENT
Whissle STT Version Release Date: May 29, 2026

"Agreement" means the terms and conditions for use, reproduction, and
distribution of the Whissle Materials set forth herein.

"Documentation" means the specifications, manuals and documentation accompanying
Whissle STT models distributed by Whissle AI at https://github.com/WhissleAI.

"Licensee" or "you" means you, or your employer or any other person or entity
(if you are entering into this Agreement on such person or entity's behalf), of
the age required under applicable laws, rules or regulations to provide legal
consent and that has legal authority to bind your employer or such other person
or entity if you are entering in this Agreement on their behalf.

"Whissle STT" means the speech recognition models distributed by Whissle AI,
including trained model weights, inference-enabling code, and associated files.

"Whissle Materials" means, collectively, Whissle AI's proprietary Whissle STT
and Documentation (and any portion thereof) made available under this Agreement.

"Whissle AI" or "we" means Whissle AI, Inc.

1. License Grant — Inference Only.

a. Grant of Rights. You are granted a non-exclusive, worldwide, non-transferable
and royalty-free limited license to use the Whissle Materials solely for
inference purposes — that is, to run the model weights to produce predictions,
transcriptions, or other outputs from input data.

b. Prohibited Uses. You shall NOT:

  i.   Use the Whissle Materials, in whole or in part, to train, fine-tune,
       adapt, or otherwise modify any machine learning or AI model, including
       the Whissle Materials themselves or any third-party model.

  ii.  Use the Whissle Materials for knowledge distillation, model compression,
       teacher-student training, or any technique that transfers learned
       representations or behaviors from the Whissle Materials into another
       model.

  iii. Reverse engineer, decompile, disassemble, or otherwise attempt to
       extract the source code, algorithms, architecture, training data,
       training methodology, or any proprietary information from the Whissle
       Materials or their outputs.

  iv.  Extract, reconstruct, or approximate the model weights, embeddings,
       or internal representations through any means, including but not
       limited to probing, model inversion, or systematic querying.

  v.   Remove, alter, or obscure any copyright, trademark, or other
       proprietary notices contained in the Whissle Materials.

c. Redistribution. If you distribute or make available the Whissle Materials
(unmodified) as part of an application or service, you shall:

  i.   Provide a copy of this Agreement with any such distribution.

  ii.  Prominently display "Powered by Whissle" on a related website, user
       interface, or product documentation.

  iii. Ensure that all recipients are bound by the terms of this Agreement,
       including the inference-only restriction.

d. Your use of the Whissle Materials must comply with applicable laws and
regulations and adhere to the Acceptable Use Policy set forth below.

2. Additional Commercial Terms. If the monthly active users of Licensee's
products or services exceeds 100 million in the preceding calendar month, you
must request a separate license from Whissle AI.

3. Disclaimer of Warranty. THE WHISSLE MATERIALS ARE PROVIDED ON AN "AS IS"
BASIS, WITHOUT WARRANTIES OF ANY KIND. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING
THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE WHISSLE MATERIALS.

4. Limitation of Liability. IN NO EVENT WILL WHISSLE AI BE LIABLE FOR ANY LOST
PROFITS OR ANY INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR
PUNITIVE DAMAGES.

5. Intellectual Property. All rights, title, and interest in the Whissle
Materials, including all intellectual property rights, remain exclusively with
Whissle AI. No rights are granted except as expressly set forth herein. If you
institute litigation alleging infringement, all licenses granted terminate.

6. Term and Termination. Whissle AI may terminate this Agreement if you breach
any term. Upon termination, you shall immediately delete all copies of the
Whissle Materials and cease all use.

7. Governing Law. This Agreement is governed by the laws of the State of Delaware.

ACCEPTABLE USE POLICY: You agree not to use the Whissle Materials to: (1) violate
applicable laws; (2) engage in surveillance or mass audio collection without
consent; (3) process minors' audio without parental consent; (4) create misleading
transcriptions; (5) discriminate based on protected characteristics; (6) deploy
in safety-critical systems without human oversight; (7) train, fine-tune, distill,
or create derivative models; (8) reverse engineer or extract proprietary information.
"""


def make_readme(model_id, info):
    languages_yaml = "\n".join(f"- {lang}" for lang in info["languages"])
    tags = "- nemo\n- asr\n- onnx\n- cpu"
    if info["head_type"] == "Dual-head":
        tags += "\n- emotion\n- age\n- gender\n- intent"

    usage_note = ""
    if info["head_type"] == "Dual-head":
        usage_note = """
The model outputs both transcription and speaker metadata via a tag classifier:
- **Age**: AGE_<20, AGE_20_30, AGE_30_45, AGE_>45
- **Gender**: MALE, FEMALE
- **Emotion**: NEUTRAL, HAPPY, SAD, ANGRY, FEAR, SURPRISE, DISGUST
- **Intent**: Various intent categories"""

    return f"""---
license: other
license_name: whissle-inference-only-1.0
license_link: LICENSE
language:
{languages_yaml}
pipeline_tag: automatic-speech-recognition
tags:
{tags}
base_model:
- {info['base_model']}
library_name: nemo
extra_gated_heading: "Access {info['name']} on Hugging Face"
extra_gated_description: >
  This model is licensed for inference only — no training, fine-tuning, distillation,
  or reverse engineering permitted. Accept the license and provide your contact information
  to access. Requests are processed automatically.
extra_gated_button_content: "Agree and access repository"
extra_gated_fields:
  First Name: text
  Last Name: text
  Organization: text
  Country: country
  Date of birth: date_picker
  I want to use this model for:
    type: select
    options:
      - Research
      - Education
      - Commercial product
      - Personal project
      - label: Other
        value: other
  I accept the Whissle Inference-Only License Agreement: checkbox
extra_gated_prompt: >-
  By clicking "Agree", you accept the Whissle Inference-Only License Agreement.
  See the LICENSE file for full terms. Key restrictions: INFERENCE ONLY — no
  training, fine-tuning, distillation, model compression, or reverse engineering
  permitted. Free for inference use under 100M MAU. "Powered by Whissle"
  attribution required for redistribution.
---

# {info['name']}

{info['description']}

- **Parameters**: {info['params']}
- **Architecture**: {info['head_type']} ({info['base_model']})
- **Languages**: {info['language_names']}
- **Format**: ONNX (CPU-optimized)

## Quick Start

Use with the [Whissle STT Inference Server](https://github.com/WhissleAI/whissle_stt_inference):

```bash
git clone https://github.com/WhissleAI/whissle_stt_inference.git
cd whissle_stt_inference
./setup.sh --model {model_id}
```

Or load directly with ONNX Runtime:

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
# Prepare mel-spectrogram input and run inference
outputs = session.run(None, {{"audio_signal": mel_features, "length": lengths}})
```
{usage_note}

## License

Licensed under the [Whissle Inference-Only License](./LICENSE). Inference only — no training, fine-tuning, distillation, or reverse engineering. Free for inference use under 100M MAU.
"""


def download_from_gcs(gcs_dir, files, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    for f in files:
        src = f"{GCS_BUCKET}/{gcs_dir}/{f}"
        dst = os.path.join(local_dir, f)
        if os.path.exists(dst):
            print(f"  {f} (cached)")
            continue
        print(f"  Downloading {f}...")
        subprocess.run(["gsutil", "cp", src, dst], check=True,
                       capture_output=True, text=True)
    print(f"  Done: {local_dir}")


def upload_to_hf(model_id, info, hf_token, cache_dir=None):
    api = HfApi(token=hf_token)
    repo_id = info["hf_repo"]

    print(f"\n{'='*60}")
    print(f"Uploading: {model_id} -> {repo_id}")
    print(f"{'='*60}")

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"  Repo: {repo_id}")
    except Exception as e:
        print(f"  Repo exists or error: {e}")

    # Download model files from GCS
    dl_dir = os.path.join(cache_dir or tempfile.mkdtemp(), model_id)
    print(f"  Downloading from GCS ({info['gcs_dir']})...")
    download_from_gcs(info["gcs_dir"], info["files"], dl_dir)

    # Generate README and LICENSE
    readme_path = os.path.join(dl_dir, "README.md")
    license_path = os.path.join(dl_dir, "LICENSE")

    with open(readme_path, "w") as f:
        f.write(make_readme(model_id, info))
    with open(license_path, "w") as f:
        f.write(WHISSLE_LICENSE)

    # Upload entire folder in one commit (much faster than file-by-file)
    total_mb = sum(
        os.path.getsize(os.path.join(dl_dir, f)) / 1024 / 1024
        for f in os.listdir(dl_dir) if os.path.isfile(os.path.join(dl_dir, f))
    )
    print(f"  Uploading folder ({total_mb:.0f} MB total)...")
    api.upload_folder(
        folder_path=dl_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add {model_id} ONNX model with Whissle Inference-Only License",
    )

    # Enable gating
    api.update_repo_settings(repo_id=repo_id, gated="auto")
    print(f"  Gating enabled (automatic approval)")

    print(f"  Done: https://huggingface.co/{repo_id}")
    return repo_id


def main():
    parser = argparse.ArgumentParser(description="Upload ONNX models to HuggingFace")
    parser.add_argument("--model", required=True,
                        help="Model ID to upload (or 'all' for all models)")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--cache-dir", default=None,
                        help="Cache directory for downloaded files")
    parser.add_argument("--list", action="store_true", help="List models")
    args = parser.parse_args()

    if args.list:
        for mid, info in MODELS.items():
            print(f"  {mid:25s} {info['hf_repo']:45s} {info['head_type']}")
        return

    if not args.hf_token:
        print("ERROR: --hf-token or HF_TOKEN env var required")
        sys.exit(1)

    if args.model == "all":
        for mid, info in MODELS.items():
            upload_to_hf(mid, info, args.hf_token, args.cache_dir)
    elif args.model in MODELS:
        upload_to_hf(args.model, MODELS[args.model], args.hf_token, args.cache_dir)
    else:
        print(f"Unknown model: {args.model}")
        print(f"Available: {', '.join(MODELS.keys())}")
        sys.exit(1)


if __name__ == "__main__":
    main()
