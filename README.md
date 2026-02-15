# HindiBabyNet-Wav2Vec2-SSL

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers 5.1](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-5.1-FFD21E)](https://huggingface.co/docs/transformers)
[![torchaudio 2.10](https://img.shields.io/badge/torchaudio-2.10-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/audio/)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-wav2vec2--home--hindibabynet--ssl-yellow)](https://huggingface.co/arunps/wav2vec2-home-hindibabynet-ssl)

Continued pretraining of `facebook/wav2vec2-base` on long-form Hindi parent–infant home recordings.

## Pretrained Model on Hugging Face

The trained model is published on the Hugging Face Hub and can be used directly for downstream tasks:

> **[arunps/wav2vec2-home-hindibabynet-ssl](https://huggingface.co/arunps/wav2vec2-home-hindibabynet-ssl)**

### Feature Extraction

```python
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import torchaudio, torch

model = Wav2Vec2Model.from_pretrained("arunps/wav2vec2-home-hindibabynet-ssl")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("arunps/wav2vec2-home-hindibabynet-ssl")

waveform, sr = torchaudio.load("your_audio.wav")
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
waveform = waveform.mean(dim=0)  # mono

inputs = feature_extractor(waveform.numpy(), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    hidden_states = model(**inputs).last_hidden_state  # (1, T, 768)
```

### Fine-tuning for ASR (CTC)

```python
from transformers import Wav2Vec2ForCTC

model = Wav2Vec2ForCTC.from_pretrained(
    "arunps/wav2vec2-home-hindibabynet-ssl",
    ctc_loss_reduction="mean",
    pad_token_id=0,
    vocab_size=YOUR_VOCAB_SIZE,
)
model.freeze_feature_encoder()
# ... fine-tune on your labelled data
```

### Fine-tuning for Audio Classification

```python
from transformers import Wav2Vec2ForSequenceClassification

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "arunps/wav2vec2-home-hindibabynet-ssl",
    num_labels=NUM_CLASSES,
)
model.freeze_feature_encoder()
# ... fine-tune on labelled classification data
```

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
