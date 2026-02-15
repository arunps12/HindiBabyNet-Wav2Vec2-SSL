"""Phase 4b — Self-supervised pretraining of HuBERT on home-domain audio.

HuBERT uses a masked-prediction objective with k-means pseudo-labels
(unlike wav2vec2's contrastive + diversity loss).

Pipeline:
  1. Run ``kmeans_labels.py`` to fit a k-means model on MFCC features.
  2. Run this script — pseudo-labels are generated on-the-fly in the collator.

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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import HubertConfig, HubertModel, Wav2Vec2FeatureExtractor

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
from homewav2vec2.train.kmeans_labels import (
    align_labels_to_model_frames,
    extract_mfcc,
    load_kmeans,
)
from homewav2vec2.utils.seed import set_seed

log = logging.getLogger("homewav2vec2.train.hubert")


# ---------------------------------------------------------------------------
# HuBERT pretraining model wrapper
# ---------------------------------------------------------------------------


class HubertForPreTraining(nn.Module):
    """HuBERT encoder + linear head for masked pseudo-label prediction.

    During pretraining the model:
      1. Receives raw audio and ``mask_time_indices``.
      2. Passes it through the CNN feature extractor + Transformer
         (masking is applied internally by ``HubertModel``).
      3. Projects hidden states at **masked** positions through a linear
         head and predicts k-means cluster IDs via cross-entropy.
    """

    def __init__(
        self,
        hubert: HubertModel,
        num_clusters: int,
        final_proj_dim: int = 256,
    ):
        super().__init__()
        self.hubert = hubert
        hidden_size = hubert.config.hidden_size
        self.final_proj = nn.Linear(hidden_size, final_proj_dim)
        self.label_head = nn.Linear(final_proj_dim, num_clusters)

    def forward(
        self,
        input_values: torch.Tensor,
        mask_time_indices: torch.BoolTensor,
        labels: torch.LongTensor | None = None,
    ):
        """Forward pass.

        Args:
            input_values: ``(B, T_audio)`` raw waveform.
            mask_time_indices: ``(B, T_frames)`` boolean mask.
            labels: ``(B, T_frames)`` k-means cluster IDs (long).

        Returns:
            Object with ``.loss`` attribute (like HF models).
        """
        outputs = self.hubert(
            input_values=input_values,
            mask_time_indices=mask_time_indices,
        )
        hidden_states = outputs.last_hidden_state  # (B, T_frames, H)

        projected = self.final_proj(hidden_states)  # (B, T_frames, proj_dim)
        logits = self.label_head(projected)  # (B, T_frames, num_clusters)

        loss = None
        if labels is not None:
            # Only compute loss at masked positions
            mask = mask_time_indices.bool()
            masked_logits = logits[mask]  # (num_masked, num_clusters)
            masked_labels = labels[mask]  # (num_masked,)
            loss = F.cross_entropy(masked_logits, masked_labels)

        return _PreTrainOutput(loss=loss, logits=logits)


class _PreTrainOutput:
    """Simple container to mimic HF model output."""
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


# ---------------------------------------------------------------------------
# Helper: compute feature-encoder output length
# ---------------------------------------------------------------------------


def _feat_extract_output_lengths(input_length: int, config: HubertConfig) -> int:
    """Compute CNN output frame count from raw waveform sample count."""
    for kernel_size, stride in zip(config.conv_kernel, config.conv_stride):
        input_length = (input_length - kernel_size) // stride + 1
    return input_length


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------


class HubertPreTrainCollator:
    """Collate crops into training batches with on-the-fly pseudo-labels.

    1. Normalise & pad waveforms via ``Wav2Vec2FeatureExtractor``.
    2. Extract MFCC features per crop → k-means assignment → pseudo-labels.
    3. Align label frame count to model output frame count.
    4. Generate ``mask_time_indices``.
    """

    def __init__(
        self,
        feature_extractor: Wav2Vec2FeatureExtractor,
        model_config: HubertConfig,
        kmeans_model,
        num_mfcc: int = 13,
        use_deltas: bool = True,
        target_sr: int = 16000,
    ):
        self.fe = feature_extractor
        self.config = model_config
        self.kmeans = kmeans_model
        self.num_mfcc = num_mfcc
        self.use_deltas = use_deltas
        self.target_sr = target_sr

    def __call__(self, batch: list[dict]) -> dict:
        waveforms = [item["input_values"] for item in batch]
        waveforms_np = [w.numpy() for w in waveforms]

        # Feature extractor normalises & pads
        features = self.fe(
            waveforms_np, sampling_rate=self.target_sr,
            return_tensors="pt", padding=True,
        )

        batch_size = features["input_values"].shape[0]
        audio_len = features["input_values"].shape[1]
        seq_len = _feat_extract_output_lengths(audio_len, self.config)

        # --- Pseudo-labels via MFCC + k-means ---
        all_labels = []
        for w in waveforms:
            mfcc_feats = extract_mfcc(w, self.target_sr, self.num_mfcc, self.use_deltas)
            cluster_ids = self.kmeans.predict(mfcc_feats)
            aligned = align_labels_to_model_frames(cluster_ids, seq_len)
            all_labels.append(aligned)
        labels = torch.from_numpy(np.stack(all_labels)).long()

        # --- Mask ---
        mask_time_indices = _compute_mask_indices(
            (batch_size, seq_len),
            mask_prob=self.config.mask_time_prob,
            mask_length=self.config.mask_time_length,
        )

        features["mask_time_indices"] = torch.from_numpy(mask_time_indices).bool()
        features["labels"] = labels
        return features


def _compute_mask_indices(shape, mask_prob=0.065, mask_length=10):
    """Simple mask generation for HuBERT pretraining (same as wav2vec2)."""
    batch_size, seq_length = shape
    mask = np.zeros(shape, dtype=bool)
    num_masked = int(mask_prob * seq_length / mask_length)

    for i in range(batch_size):
        starts = np.random.choice(
            max(seq_length - mask_length, 1),
            size=max(num_masked, 1),
            replace=False,
        )
        for start in starts:
            mask[i, start: start + mask_length] = True
    return mask


# ---------------------------------------------------------------------------
# Config loading (reused from wav2vec2 with HuBERT defaults)
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_configs(training_cfg_path: str, data_cfg_path: str | None = None) -> dict:
    cfg = load_config(training_cfg_path)
    if data_cfg_path:
        data_cfg = load_config(data_cfg_path)
        cfg.update({k: v for k, v in data_cfg.items() if k not in cfg})
    defaults_path = Path(training_cfg_path).parent / "defaults.yaml"
    if defaults_path.exists():
        defaults = load_config(str(defaults_path))
        for k, v in defaults.items():
            cfg.setdefault(k, v)
    return cfg


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(cfg: dict) -> None:
    """Core HuBERT pretraining function."""
    set_seed(cfg.get("seed", 42))
    setup_distributed()

    device = torch.device(f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu")

    # === Load k-means model ===
    kmeans_dir = Path(cfg.get("kmeans_dir", "artifacts/hubert_training/kmeans"))
    kmeans_path = kmeans_dir / "kmeans_model.pkl"
    if not kmeans_path.exists():
        raise FileNotFoundError(
            f"K-means model not found at {kmeans_path}. "
            "Run `python -m homewav2vec2.train.kmeans_labels` first."
        )
    kmeans_model = load_kmeans(kmeans_path)
    num_clusters = kmeans_model.n_clusters
    log.info("Loaded k-means model with %d clusters from %s", num_clusters, kmeans_path)

    # === Model ===
    model_name = cfg.get("base_model", "facebook/hubert-base-ls960")
    config = HubertConfig.from_pretrained(model_name)
    hubert_base = HubertModel.from_pretrained(model_name, config=config)

    model = HubertForPreTraining(
        hubert=hubert_base,
        num_clusters=num_clusters,
        final_proj_dim=cfg.get("final_proj_dim", 256),
    )

    if cfg.get("gradient_checkpointing", True):
        model.hubert.gradient_checkpointing_enable()

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
        epoch_multiplier=cfg.get("epoch_multiplier", 1),
    )

    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed() else None
    raw_model = model.module if is_distributed() else model
    collator = HubertPreTrainCollator(
        feature_extractor=fe,
        model_config=raw_model.hubert.config,
        kmeans_model=kmeans_model,
        num_mfcc=cfg.get("num_mfcc", 13),
        use_deltas=cfg.get("use_deltas", True),
        target_sr=cfg.get("sample_rate", 16000),
    )

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
    scaler = GradScaler("cuda", enabled=use_fp16)

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
    output_dir = Path(cfg.get("output_dir", "artifacts/hubert_training"))
    ckpt_dir = Path(cfg.get("checkpoint_dir", "artifacts/hubert_training/checkpoints"))
    log_dir = Path(cfg.get("log_dir", "artifacts/hubert_training/logs"))

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
    log.info(
        "HuBERT training: model=%s clusters=%d epochs=%d total_steps=%d "
        "fp16=%s grad_accum=%d",
        model_name, num_clusters, num_epochs, total_steps, use_fp16, grad_accum,
    )

    model.train()

    pbar = tqdm(
        total=total_steps,
        initial=global_step,
        desc="HuBERT SSL",
        unit="step",
        disable=not is_main_process(),
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    max_epochs = num_epochs if max_steps <= 0 else 10_000_000
    epoch = start_epoch
    done = False

    for epoch in range(start_epoch, max_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        accumulation_loss = 0.0
        for step_in_epoch, batch in enumerate(dataloader):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with autocast("cuda", enabled=use_fp16):
                outputs = model(
                    input_values=batch["input_values"],
                    mask_time_indices=batch["mask_time_indices"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / grad_accum

            scaler.scale(loss).backward()
            accumulation_loss += loss.item()

            if (step_in_epoch + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.get("max_grad_norm", 1.0)
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                lr_current = scheduler.get_last_lr()[0]
                current_loss = accumulation_loss * grad_accum
                pbar.update(1)
                pbar.set_postfix(
                    epoch=epoch,
                    loss=f"{current_loss:.4f}",
                    lr=f"{lr_current:.2e}",
                )

                if is_main_process() and global_step % logging_steps == 0:
                    log.info(
                        "epoch=%d step=%d loss=%.4f lr=%.2e",
                        epoch, global_step, current_loss, lr_current,
                    )
                    if metrics_logger:
                        metrics_logger.log(global_step, {
                            "epoch": epoch,
                            "loss": round(current_loss, 4),
                            "lr": lr_current,
                        })
                    accumulation_loss = 0.0

                if is_main_process() and global_step % save_steps == 0:
                    _save_checkpoint(
                        raw_model, optimizer, scheduler, scaler, use_fp16,
                        global_step, epoch, ckpt_dir, save_total_limit,
                    )

                if max_steps > 0 and global_step >= max_steps:
                    done = True
                    break

        if done:
            break

    pbar.close()

    # === Save final model ===
    if is_main_process():
        final_dir = output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)

        # Save HuBERT encoder (without the pretraining head) for downstream use
        raw_model.hubert.save_pretrained(str(final_dir))
        fe.save_pretrained(str(final_dir))

        # Also save the full pretraining model (with head) separately
        full_dir = output_dir / "final_model_with_head"
        full_dir.mkdir(parents=True, exist_ok=True)
        torch.save(raw_model.state_dict(), full_dir / "hubert_pretrain_full.pt")

        log.info("Final HuBERT encoder saved to %s", final_dir)
        log.info("Full pretraining model saved to %s", full_dir)

        state_path = output_dir / "trainer_state.json"
        state_path.write_text(json.dumps({
            "global_step": global_step,
            "epochs_completed": epoch + 1,
            "final_model_dir": str(final_dir),
            "num_clusters": num_clusters,
            "base_model": model_name,
        }, indent=2))

        if metrics_logger:
            metrics_logger.close()

    cleanup_distributed()
    log.info("HuBERT training complete — step %d", global_step)


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
    model.hubert.save_pretrained(str(ckpt_path))
    log.info("Checkpoint saved: %s", ckpt_path)

    existing = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    while len(existing) > save_total_limit:
        oldest = existing.pop(0)
        import shutil
        shutil.rmtree(oldest)
        log.info("Pruned old checkpoint: %s", oldest)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="HuBERT SSL pretraining")
    parser.add_argument("--config", required=True, help="Path to training_hubert.yaml")
    parser.add_argument("--data-config", default=None, help="Path to data.yaml")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Checkpoint path")
    args = parser.parse_args()

    from homewav2vec2.utils.logging import setup_logging
    setup_logging("INFO")

    cfg = merge_configs(args.config, args.data_config)
    if args.resume_from_checkpoint:
        cfg["resume_from_checkpoint"] = args.resume_from_checkpoint

    train(cfg)


if __name__ == "__main__":
    main()
