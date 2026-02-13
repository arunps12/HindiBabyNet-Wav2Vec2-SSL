# HindiBabyNet-Wav2Vec2-SSL

Continued pretraining of `facebook/wav2vec2-base` on long-form Hindi parent–infant home recordings.

## Quick Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/arunps12/HindiBabyNet-Wav2Vec2-SSL.git
cd HindiBabyNet-Wav2Vec2-SSL

# 2. Create a virtual environment and install dependencies
uv venv
uv sync

# 3. Configure secrets
cp .env.example .env
# Edit .env and fill in HF_TOKEN
```

## Pipeline Stages

### Phase 1 — Data Ingestion (manifest only)

Scans raw audio and creates a metadata manifest (no audio is copied).

```bash
bash scripts/run_ingestion.sh
# or:
python -m homewav2vec2.cli ingest --config src/homewav2vec2/config/data.yaml
```

**Output:** `artifacts/manifests/raw_audio_manifest.parquet`

### Phase 2 — Data Validation

Checks every file for readability, duration, silence, etc. Does NOT delete anything.

```bash
bash scripts/run_validation.sh
# or:
python -m homewav2vec2.cli validate --config src/homewav2vec2/config/data.yaml
```

**Output:**
- `artifacts/validation/invalid_files.parquet`
- `artifacts/validation/summary.json`
- `artifacts/stats/audio_stats.parquet`

### Phase 3 — Transformation (splits, NO chunk files)

Creates train/dev splits by participant ID. Audio is NEVER saved to disk — random cropping happens on-the-fly during training.

```bash
bash scripts/run_transformation.sh
# or:
python -m homewav2vec2.cli transform --config src/homewav2vec2/config/data.yaml
```

**Output:**
- `artifacts/manifests/train_split.parquet`
- `artifacts/manifests/dev_split.parquet`
- `artifacts/stats/split_summary.json`

### Phase 4 — SSL Training

Self-supervised pretraining with `Wav2Vec2ForPreTraining`.

```bash
# Single GPU
bash scripts/run_train_ssl.sh

# Multi-GPU (4 GPUs)
bash scripts/run_train_ssl.sh 4

# Or directly with torchrun:
torchrun --nproc_per_node 4 \
    -m homewav2vec2.train.run_pretrain \
    --config src/homewav2vec2/config/training.yaml \
    --data-config src/homewav2vec2/config/data.yaml
```

**Default config for RTX 2080 Ti (11 GB):**
- `crop_sec=8`, `per_gpu_batch=2`, `grad_accum_steps=8`
- `fp16=true`, `gradient_checkpointing=true`

**Output:** `artifacts/training/` (checkpoints, logs, final model)

### Phase 5 — Push to Hugging Face Hub

```bash
bash scripts/run_push_hf.sh
# or:
python -m homewav2vec2.cli push_hf --config src/homewav2vec2/config/training.yaml
```

Requires `HF_TOKEN` in `.env`. Never prints the token.

## Configuration

- `src/homewav2vec2/config/data.yaml` — raw audio root, sample rate, cropping params, splits
- `src/homewav2vec2/config/training.yaml` — model, hyperparams, checkpointing
- `src/homewav2vec2/config/defaults.yaml` — seed, num_workers, log level
- `.env` — secrets (HF_TOKEN, HF_REPO_ID)

## Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
src/homewav2vec2/
├── __init__.py / __main__.py / cli.py
├── config/          — YAML configs
├── data/            — ingest, validate, transform, manifests
├── dataset/         — audio_io, cropping, hf_dataset (on-the-fly)
├── train/           — run_pretrain, ddp, callbacks
└── utils/           — env, logging, seed
scripts/             — shell wrappers
artifacts/           — manifests, stats, validation, training outputs
tests/               — pytest suite
```

## Key Design Decisions

1. **No chunk WAVs saved to disk** — training uses on-the-fly random cropping with partial file reads.
2. **Raw audio stays external** at `/scratch/users/arunps/hindibabynet/audio_raw/RawAudioData`.
3. **uv** for dependency management — `uv venv && uv sync` for reproducible setup.
4. **fp16 + gradient checkpointing** by default for 11 GB GPU memory.

## License

See [LICENSE](LICENSE).
