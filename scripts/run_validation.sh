#!/usr/bin/env bash
# Phase 2: Validate audio files
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

python -m homewav2vec2.cli validate --config src/homewav2vec2/config/data.yaml
