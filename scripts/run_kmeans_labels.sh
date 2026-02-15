#!/usr/bin/env bash
# Fit k-means on MFCC features for HuBERT pseudo-labels
# Must run AFTER ingestion/validation/transformation (needs train_split.parquet)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="src/homewav2vec2/config/training_hubert.yaml"
DATA_CONFIG="src/homewav2vec2/config/data.yaml"

python -m homewav2vec2.train.kmeans_labels \
    --config "$CONFIG" \
    --data-config "$DATA_CONFIG"
