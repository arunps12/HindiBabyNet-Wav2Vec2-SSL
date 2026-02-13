#!/usr/bin/env bash
# Phase 5: Push final model to Hugging Face Hub
# Requires .env with HF_TOKEN set
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

python -m homewav2vec2.cli push_hf --config src/homewav2vec2/config/training.yaml
