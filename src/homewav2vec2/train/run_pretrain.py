"""Phase 4 — Self-supervised pretraining of Wav2Vec2 on home-domain audio.

Supports:
  - single GPU
  - multi-GPU via ``torchrun`` (DDP)
  - fp16 mixed precision
  - gradient checkpointing
  - checkpoint resume
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import Wav2Vec2Config, Wav2Vec2ForPreTraining, Wav2Vec2FeatureExtractor

from homewav2vec2.dataset.hf_dataset import AudioCropDataset
from homewav2vec2.train.callbacks import MetricsLogger
from homewav2vec2.train.ddp import (
    cleanup_distributed,
    global_rank,
    is_distributed,
    is_main_process,
    local_rank,
    setup_distributed,
    world_size,
)
from homewav2vec2.utils.seed import set_seed

log = logging.getLogger("homewav2vec2.train")


# ---------------------------------------------------------------------------
# Data collator for Wav2Vec2ForPreTraining
# ---------------------------------------------------------------------------

class Wav2Vec2PreTrainCollator:
    """Collate cropped waveforms into batch tensors expected by
    ``Wav2Vec2ForPreTraining``.

    The feature extractor normalises and pads waveforms.  We create the
    ``mask_time_indices`` and ``sampled_negative_indices`` required by the
    contrastive + diversity SSL loss.
    """

    def __init__(self, feature_extractor: Wav2Vec2FeatureExtractor, model_config: Wav2Vec2Config):
        self.fe = feature_extractor
        self.config = model_config

    def __call__(self, batch: list[dict]) -> dict:
        waveforms = [item["input_values"].numpy() for item in batch]

        # Feature extractor normalises & pads
        features = self.fe(
            waveforms, sampling_rate=16000, return_tensors="pt", padding=True,
        )

        batch_size = features["input_values"].shape[0]
        seq_len = self.config._get_feat_extract_output_lengths(features["input_values"].shape[1])
        if isinstance(seq_len, torch.Tensor):
            seq_len = seq_len.item()

        # Create mask_time_indices
        mask_time_indices = _compute_mask_indices(
            (batch_size, seq_len),
            mask_prob=self.config.mask_time_prob,
            mask_length=self.config.mask_time_length,
        )
        # Sampled negatives
        sampled_negative_indices = _sample_negative_indices(
            (batch_size, seq_len),
            num_negatives=self.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )

        features["mask_time_indices"] = torch.from_numpy(mask_time_indices)
        features["sampled_negative_indices"] = torch.from_numpy(sampled_negative_indices)
        return features


def _compute_mask_indices(shape, mask_prob=0.065, mask_length=10):
    """Simple mask generation for wav2vec2 pretraining."""
    import numpy as np

    batch_size, seq_length = shape
    mask = np.zeros(shape, dtype=bool)
    num_masked = int(mask_prob * seq_length / mask_length)

    for i in range(batch_size):
        starts = np.random.choice(max(seq_length - mask_length, 1), size=max(num_masked, 1), replace=False)
        for start in starts:
            mask[i, start: start + mask_length] = True
    return mask


def _sample_negative_indices(features_shape, num_negatives, mask_time_indices):
    """Sample negative indices for contrastive loss."""
    import numpy as np

    batch_size, seq_length = features_shape
    sampled = np.zeros((batch_size, seq_length, num_negatives), dtype=np.int32)

    for i in range(batch_size):
        masked_idx = np.where(mask_time_indices[i])[0]
        if len(masked_idx) == 0:
            continue
        for j in range(seq_length):
            candidates = masked_idx[masked_idx != j] if j in masked_idx else masked_idx
            if len(candidates) == 0:
                candidates = masked_idx
            chosen = np.random.choice(candidates, size=num_negatives, replace=True)
            sampled[i, j] = chosen

    return sampled


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_configs(training_cfg_path: str, data_cfg_path: str | None = None) -> dict:
    """Merge training.yaml (+ optional data.yaml) into a single dict."""
    cfg = load_config(training_cfg_path)
    if data_cfg_path:
        data_cfg = load_config(data_cfg_path)
        cfg.update({k: v for k, v in data_cfg.items() if k not in cfg})
    # Load defaults.yaml if present next to training.yaml
    defaults_path = Path(training_cfg_path).parent / "defaults.yaml"
    if defaults_path.exists():
        defaults = load_config(str(defaults_path))
        for k, v in defaults.items():
            cfg.setdefault(k, v)
    return cfg


def train(cfg: dict) -> None:
    """Core training function."""
    set_seed(cfg.get("seed", 42))
    setup_distributed()

    device = torch.device(f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu")

    # === Model ===
    model_name = cfg.get("base_model", "facebook/wav2vec2-base")
    config = Wav2Vec2Config.from_pretrained(model_name)
    model = Wav2Vec2ForPreTraining.from_pretrained(model_name, config=config)

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    model.to(device)

    if is_distributed():
        model = DDP(model, device_ids=[local_rank()], output_device=local_rank())

    # === Feature extractor ===
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    # === Dataset ===
    artifacts_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    train_manifest = artifacts_dir / "manifests" / "train_split.parquet"

    dataset = AudioCropDataset(
        manifest_path=train_manifest,
        crop_sec=cfg.get("crop_sec", 8.0),
        target_sr=cfg.get("sample_rate", 16000),
        silence_rms_threshold=cfg.get("silence_rms_threshold", 0.001),
        max_crop_retries=cfg.get("max_crop_retries", 10),
    )

    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed() else None
    raw_model = model.module if is_distributed() else model
    collator = Wav2Vec2PreTrainCollator(fe, raw_model.config)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("per_gpu_batch", 2),
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    # === Optimiser ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("learning_rate", 5e-5),
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    # === AMP scaler ===
    use_fp16 = cfg.get("fp16", True) and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_fp16)

    # === LR scheduler ===
    num_epochs = cfg.get("num_train_epochs", 10)
    max_steps = cfg.get("max_steps", -1)
    warmup_steps = cfg.get("warmup_steps", 500)
    grad_accum = cfg.get("grad_accum_steps", 8)
    steps_per_epoch = math.ceil(len(dataloader) / grad_accum)
    total_steps = max_steps if max_steps > 0 else steps_per_epoch * num_epochs

    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - current_step) / max(1, total_steps - warmup_steps))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # === Logging / checkpointing ===
    output_dir = Path(cfg.get("output_dir", "artifacts/training"))
    ckpt_dir = Path(cfg.get("checkpoint_dir", "artifacts/training/checkpoints"))
    log_dir = Path(cfg.get("log_dir", "artifacts/training/logs"))

    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    metrics_logger = MetricsLogger(log_dir) if is_main_process() else None
    logging_steps = cfg.get("logging_steps", 50)
    save_steps = cfg.get("save_steps", 1000)
    save_total_limit = cfg.get("save_total_limit", 3)

    # === Resume from checkpoint ===
    global_step = 0
    start_epoch = 0
    resume_ckpt = cfg.get("resume_from_checkpoint")
    if resume_ckpt and Path(resume_ckpt).exists():
        ckpt = torch.load(resume_ckpt, map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if use_fp16 and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        log.info("Resumed from checkpoint: step=%d epoch=%d", global_step, start_epoch)

    # === Training loop ===
    log.info("Training config: epochs=%d total_steps=%d fp16=%s grad_accum=%d",
             num_epochs, total_steps, use_fp16, grad_accum)

    model.train()
    for epoch in range(start_epoch, num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        accumulation_loss = 0.0
        for step_in_epoch, batch in enumerate(dataloader):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with autocast(enabled=use_fp16):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum

            scaler.scale(loss).backward()
            accumulation_loss += loss.item()

            if (step_in_epoch + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                # Logging
                if is_main_process() and global_step % logging_steps == 0:
                    lr_current = scheduler.get_last_lr()[0]
                    log.info(
                        "epoch=%d step=%d loss=%.4f lr=%.2e",
                        epoch, global_step, accumulation_loss * grad_accum, lr_current,
                    )
                    if metrics_logger:
                        metrics_logger.log(global_step, {
                            "epoch": epoch,
                            "loss": round(accumulation_loss * grad_accum, 4),
                            "lr": lr_current,
                        })
                    accumulation_loss = 0.0

                # Save checkpoint
                if is_main_process() and global_step % save_steps == 0:
                    _save_checkpoint(
                        raw_model, optimizer, scheduler, scaler, use_fp16,
                        global_step, epoch, ckpt_dir, save_total_limit,
                    )

                if max_steps > 0 and global_step >= max_steps:
                    break

        if max_steps > 0 and global_step >= max_steps:
            break

    # === Save final model ===
    if is_main_process():
        final_dir = output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        raw_model.save_pretrained(str(final_dir))
        fe.save_pretrained(str(final_dir))
        log.info("Final model saved to %s", final_dir)

        # Trainer state summary
        state_path = output_dir / "trainer_state.json"
        state_path.write_text(json.dumps({
            "global_step": global_step,
            "epochs_completed": num_epochs,
            "final_model_dir": str(final_dir),
        }, indent=2))

        if metrics_logger:
            metrics_logger.close()

    cleanup_distributed()
    log.info("Training complete — step %d", global_step)


def _save_checkpoint(model, optimizer, scheduler, scaler, use_fp16,
                     step, epoch, ckpt_dir, save_total_limit):
    """Save a training checkpoint, pruning old ones if over limit."""
    ckpt_path = ckpt_dir / f"checkpoint-{step}"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": step,
        "epoch": epoch,
    }
    if use_fp16:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, ckpt_path / "training_state.pt")
    model.save_pretrained(str(ckpt_path))
    log.info("Checkpoint saved: %s", ckpt_path)

    # Prune old checkpoints
    existing = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    while len(existing) > save_total_limit:
        oldest = existing.pop(0)
        import shutil
        shutil.rmtree(oldest)
        log.info("Pruned old checkpoint: %s", oldest)


# ---------------------------------------------------------------------------
# CLI entry — can be invoked directly or via torchrun
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Wav2Vec2 SSL pretraining")
    parser.add_argument("--config", required=True, help="Path to training.yaml")
    parser.add_argument("--data-config", default=None, help="Path to data.yaml (optional)")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Checkpoint path to resume")
    args = parser.parse_args()

    from homewav2vec2.utils.logging import setup_logging
    setup_logging("INFO")

    cfg = merge_configs(args.config, args.data_config)
    if args.resume_from_checkpoint:
        cfg["resume_from_checkpoint"] = args.resume_from_checkpoint

    train(cfg)


if __name__ == "__main__":
    main()
