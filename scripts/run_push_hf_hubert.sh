#!/usr/bin/env bash
# Push HuBERT model to Hugging Face Hub
# Requires .env with HF_TOKEN and HF_HUBERT_REPO_ID set
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

python -m homewav2vec2.cli push_hf_hubert --config src/homewav2vec2/config/training_hubert.yaml
