"""Hugging Face compatible dataset that performs on-the-fly random cropping."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from homewav2vec2.data.manifests import read_manifest
from homewav2vec2.dataset.cropping import random_crop

log = logging.getLogger("homewav2vec2.dataset")


class AudioCropDataset(Dataset):
    """PyTorch Dataset that yields random crops from long recordings.

    Each ``__getitem__`` call picks a random window — no chunk files are
    ever saved to disk.  Compatible with ``DataLoader`` + DDP sampler.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        crop_sec: float = 8.0,
        target_sr: int = 16000,
        silence_rms_threshold: float = 0.001,
        max_crop_retries: int = 10,
        epoch_multiplier: int = 1,
        seed: int = 42,
    ):
        self.df = read_manifest(manifest_path)
        self.crop_sec = crop_sec
        self.target_sr = target_sr
        self.silence_rms_threshold = silence_rms_threshold
        self.max_crop_retries = max_crop_retries
        self.epoch_multiplier = epoch_multiplier
        self.seed = seed

        log.info(
            "AudioCropDataset: %d files, crop=%.1fs, multiplier=%d",
            len(self.df), crop_sec, epoch_multiplier,
        )

    def __len__(self) -> int:
        return len(self.df) * self.epoch_multiplier

    def __getitem__(self, idx: int) -> dict[str, Any]:
        real_idx = idx % len(self.df)
        row = self.df.iloc[real_idx]

        # Per-sample RNG so each worker produces different crops
        rng = np.random.RandomState(self.seed + idx)

        waveform = random_crop(
            wav_path=row["wav_path"],
            crop_sec=self.crop_sec,
            target_sr=self.target_sr,
            silence_rms_threshold=self.silence_rms_threshold,
            max_retries=self.max_crop_retries,
            rng=rng,
        )

        return {
            "input_values": torch.from_numpy(waveform).float(),
            "wav_path": row["wav_path"],
        }


def build_hf_dataset_from_manifest(
    manifest_path: str | Path,
    crop_sec: float = 8.0,
    target_sr: int = 16000,
    **kwargs: Any,
) -> AudioCropDataset:
    """Convenience constructor."""
    return AudioCropDataset(
        manifest_path=manifest_path,
        crop_sec=crop_sec,
        target_sr=target_sr,
        **kwargs,
    )
