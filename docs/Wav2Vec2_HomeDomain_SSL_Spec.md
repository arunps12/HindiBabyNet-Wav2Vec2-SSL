# Wav2Vec2 Home-Domain Self-Supervised Training Spec (HindiBabyNet)

Generated on: 2026-02-13

------------------------------------------------------------------------

# OBJECTIVE

Implement a **research-grade** (not production packaging) pipeline to run **self-supervised pretraining / continued pretraining**
of **facebook/wav2vec2-base** on **long-form home parent–infant interaction recordings**.

Constraints / requirements:

1. **Hugging Face** training workflow (Transformers + Datasets)
2. **No chunk files saved to disk** (use on-the-fly random cropping from long WAVs)
3. Data lives outside the repo:

   - Raw audio root:
     `/scratch/users/arunps/hindibabynet/audio_raw/RawAudioData`

4. Provide **data ingestion, validation, transformation** stages (manifests + metadata only)
5. Train with **multi-GPU** (DDP) on UiO ML nodes (RTX 2080 Ti, 11GB)
6. Use `.env` for secrets (HF token), and **push model to Hugging Face Hub**
7. Keep overall structure **similar** to the attached “spec-style” documents, but **no production server / monitoring / IaC**

This document is the **SINGLE SOURCE OF TRUTH** for the code agent.

------------------------------------------------------------------------

# PHASE 0 — STRUCTURE AUDIT (MANDATORY FIRST STEP)

1. Inspect the full repository tree.
2. Identify:
   - dead/unused scripts and notebooks
   - redundant folders
   - duplicate logic
   - hardcoded paths (especially paths that should be configurable)
3. Propose minimal deletions/moves, then execute them.
4. Preserve working behavior where it exists.
5. Every phase must end with a **git commit** (small, incremental commits).

------------------------------------------------------------------------

# TARGET REPOSITORY STRUCTURE

ProjectRoot/
├── src/
│   └── homewav2vec2/
│       ├── __init__.py
│       ├── config/
│       │   ├── defaults.yaml
│       │   ├── training.yaml
│       │   └── data.yaml
│       ├── data/
│       │   ├── ingest.py
│       │   ├── validate.py
│       │   ├── transform.py
│       │   └── manifests.py
│       ├── dataset/
│       │   ├── hf_dataset.py
│       │   ├── cropping.py
│       │   └── audio_io.py
│       ├── train/
│       │   ├── run_pretrain.py
│       │   ├── ddp.py
│       │   └── callbacks.py
│       ├── utils/
│       │   ├── env.py
│       │   ├── logging.py
│       │   └── seed.py
│       └── cli.py
├── scripts/
│   ├── run_ingestion.sh
│   ├── run_validation.sh
│   ├── run_transformation.sh
│   ├── run_train_ssl.sh
│   └── run_push_hf.sh
├── artifacts/
│   ├── manifests/
│   ├── validation/
│   ├── stats/
│   └── training/
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── dvc.yaml   (optional but recommended for stage reproducibility)

Notes:
- Keep code importable from `src/`.
- Do NOT copy raw audio into the repo.
- Artifacts are **metadata only** (manifests, stats, logs, checkpoints symlinks, etc.).
- If DVC is used: track artifacts and pipeline stages; do NOT track raw audio.

------------------------------------------------------------------------

# CONFIGURATION & SECRETS

## .env

Create `.env.example` (do not commit `.env`):

HF_TOKEN=YOUR_TOKEN_HERE
HF_REPO_ID=arunps/wav2vec2-home-hindibabynet   # change as needed
HF_PRIVATE=true                                # true/false

Optionally:
WANDB_API_KEY=...
HF_HOME=/scratch/users/arunps/.cache/huggingface
HF_DATASETS_CACHE=/scratch/users/arunps/.cache/huggingface/datasets
TRANSFORMERS_CACHE=/scratch/users/arunps/.cache/huggingface/hub

## Config files

- `data.yaml`:
  - raw_audio_root: `/scratch/users/arunps/hindibabynet/audio_raw/RawAudioData`
  - allowed_ext: [".wav", ".flac"]
  - sample_rate: 16000
  - min_duration_sec: 2.0
  - max_duration_sec: null
  - channel: "mono" (convert on-the-fly)
- `training.yaml`:
  - base_model: facebook/wav2vec2-base
  - crop_sec: 8.0   (default for 11GB GPUs)
  - per_gpu_batch: 2
  - grad_accum_steps: 8
  - fp16: true
  - gradient_checkpointing: true
  - max_steps / num_train_epochs
  - learning_rate, warmup_steps, weight_decay
  - save_steps, logging_steps, eval strategy (optional)
- `defaults.yaml`: global seed, num_workers, etc.

------------------------------------------------------------------------

# PHASE 1 — DATA INGESTION (MANIFEST-ONLY)

Goal: create a deterministic manifest of available raw audio without touching/copying it.

Create:
- `artifacts/manifests/raw_audio_manifest.parquet` (or .csv)
- `artifacts/manifests/raw_audio_manifest.jsonl` (optional)

Manifest columns (minimum):
- participant_id (if derivable from folder / filename; else "unknown")
- recording_id  (filename stem)
- wav_path      (absolute path)
- duration_sec  (computed)
- sample_rate   (detected)
- channels      (detected)
- file_size_bytes
- sha1 (optional; expensive; allow switch)

Rules:
- Must be deterministic ordering.
- Must never modify raw audio.
- Must support incremental scans (only re-check changed files if possible).

Script entrypoint:
- `python -m homewav2vec2.cli ingest --config configs/data.yaml`

`scripts/run_ingestion.sh` wraps the command.

Git commit at end:
- "stage1: manifest-only ingestion"

------------------------------------------------------------------------

# PHASE 2 — DATA VALIDATION

Goal: detect problematic files and record quality stats.

Outputs (metadata only):
- `artifacts/validation/invalid_files.parquet`
- `artifacts/validation/summary.json`
- `artifacts/stats/audio_stats.parquet`

Validation checks:
- readable by soundfile/torchaudio
- duration_sec >= min_duration_sec
- sample rate sanity (allow resample later, but report)
- empty/near-silent detection (RMS threshold) — report, do not delete
- clipping rate (optional; report)
- channels: if >1, note conversion policy

Do NOT delete files; only record invalid list and reasons.

CLI:
- `python -m homewav2vec2.cli validate --config configs/data.yaml`

Git commit:
- "stage2: validation reports"

------------------------------------------------------------------------

# PHASE 3 — DATA TRANSFORMATION (NO SAVED CHUNKS)

Goal: create a Hugging Face dataset (or dataset builder) that performs:
- resampling to 16 kHz
- mono conversion
- **random cropping** (fixed length) on-the-fly
- optional silence rejection for crops (reject and re-sample)

Key rule:
- **Never write chunk wav files** to the project folder or artifacts.

Implementation approach:

A) Manifest-based dataset
- Use manifest entries as “records”
- In `__getitem__`, pick a random start time and read only that window.

B) Efficient partial read
- Prefer `soundfile.SoundFile(..., start=..., frames=...)` or torchaudio frame offsets
- Avoid loading the entire 6-hour file into memory

Cropping policy (default):
- crop_sec = 8.0 (configurable)
- attempt up to N=10 resamples if crop is “too silent”
- if still silent, accept (or mark) based on config

Outputs:
- `artifacts/manifests/train_split.parquet`
- `artifacts/manifests/dev_split.parquet` (optional)
- `artifacts/stats/split_summary.json`

Split policy:
- by participant_id (preferred) to avoid leakage
- or by recording_id if participant_id not available

CLI:
- `python -m homewav2vec2.cli transform --config configs/data.yaml`

Git commit:
- "stage3: on-the-fly dataset + splits"

------------------------------------------------------------------------

# PHASE 4 — SELF-SUPERVISED TRAINING (HUGGING FACE)

Goal: continued pretraining of `facebook/wav2vec2-base` on home-domain audio.

Training script:
- `src/homewav2vec2/train/run_pretrain.py`

Must support:
- single GPU
- multi-GPU DDP (torchrun)
- fp16
- gradient checkpointing
- resuming from checkpoint
- logging to artifacts
- saving model + processor configs

Recommended baseline for RTX 2080 Ti (11GB):
- crop_sec: 8.0
- per_gpu_batch: 2
- grad_accum_steps: 8
- fp16: true
- gradient_checkpointing: true
- num_workers: 4–8
- learning_rate: 5e-5 to 1e-4 (tune)

Example launch:
- `torchrun --nproc_per_node 4 -m homewav2vec2.train.run_pretrain --config configs/training.yaml`

Artifacts:
- `artifacts/training/checkpoints/`
- `artifacts/training/logs/`
- `artifacts/training/trainer_state.json`

Git commit:
- "stage4: wav2vec2 ssl training pipeline"

------------------------------------------------------------------------

# PHASE 5 — PUSH TO HUGGING FACE HUB

Goal: push final model (and optionally checkpoints) to the Hub using token from `.env`.

Requirements:
- Load `.env` safely
- Never print HF_TOKEN
- Support private repos (HF_PRIVATE=true)
- Use `huggingface_hub` and/or `transformers.Trainer.push_to_hub`

Push contents:
- model weights
- config.json
- preprocessor_config.json (feature extractor)
- README / model card (minimal)

CLI:
- `python -m homewav2vec2.cli push_hf --config configs/training.yaml`

Also provide a shell wrapper:
- `scripts/run_push_hf.sh`

Git commit:
- "stage5: hf hub push"

------------------------------------------------------------------------

# OPTIONAL — DVC PIPELINE (REPRODUCIBLE STAGES)

If using DVC, create `dvc.yaml` with stages:

stages:
  ingestion:
    cmd: bash scripts/run_ingestion.sh
    outs:
      - artifacts/manifests/raw_audio_manifest.parquet
  validation:
    cmd: bash scripts/run_validation.sh
    deps:
      - artifacts/manifests/raw_audio_manifest.parquet
    outs:
      - artifacts/validation/summary.json
      - artifacts/validation/invalid_files.parquet
  transformation:
    cmd: bash scripts/run_transformation.sh
    deps:
      - artifacts/manifests/raw_audio_manifest.parquet
      - artifacts/validation/invalid_files.parquet
    outs:
      - artifacts/manifests/train_split.parquet
      - artifacts/manifests/dev_split.parquet
  training:
    cmd: bash scripts/run_train_ssl.sh
    deps:
      - artifacts/manifests/train_split.parquet
    outs:
      - artifacts/training/

Note:
- Do not add raw audio to DVC.
- DVC is for manifests/stats/checkpoints only.

------------------------------------------------------------------------

# TESTS (MINIMUM REQUIRED)

Add tests to ensure:
- manifest generation is deterministic
- validation flags known-bad inputs
- dataset cropping returns correct length (samples == crop_sec * sr)
- silence rejection logic terminates
- training script parses config and can run a tiny smoke step on CPU

------------------------------------------------------------------------

# DEFINITION OF DONE

- Repo has clean src-layout structure as specified
- Ingestion/validation/transformation run end-to-end without creating chunk wav files
- Training runs on 1+ GPUs with fp16 + checkpointing
- Final model is pushed to HF Hub using `.env` token
- Artifacts are stored under `artifacts/` (metadata, logs, checkpoints)
- Each phase is committed in git with clear messages

End of specification.
