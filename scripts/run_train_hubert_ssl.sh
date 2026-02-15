#!/usr/bin/env bash
# HuBERT SSL pretraining
# Single GPU:  bash scripts/run_train_hubert_ssl.sh
# Multi-GPU:   bash scripts/run_train_hubert_ssl.sh 4    (for 4 GPUs)
#
# Prerequisites:
#   1. bash scripts/run_ingestion.sh
#   2. bash scripts/run_validation.sh
#   3. bash scripts/run_transformation.sh
#   4. bash scripts/run_kmeans_labels.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

NUM_GPUS="${1:-1}"
CONFIG="src/homewav2vec2/config/training_hubert.yaml"
DATA_CONFIG="src/homewav2vec2/config/data.yaml"

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node "$NUM_GPUS" \
        -m homewav2vec2.train.run_pretrain_hubert \
        --config "$CONFIG" \
        --data-config "$DATA_CONFIG"
else
    python -m homewav2vec2.train.run_pretrain_hubert \
        --config "$CONFIG" \
        --data-config "$DATA_CONFIG"
fi
