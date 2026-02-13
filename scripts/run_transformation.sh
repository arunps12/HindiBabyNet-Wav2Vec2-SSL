#!/usr/bin/env bash
# Phase 3: Create train/dev splits (manifest-only, NO chunk files)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

python -m homewav2vec2.cli transform --config src/homewav2vec2/config/data.yaml
