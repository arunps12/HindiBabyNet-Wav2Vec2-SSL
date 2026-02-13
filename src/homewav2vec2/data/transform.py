"""Phase 3 — Transformation: create train/dev splits (manifest-only, NO chunk files)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from homewav2vec2.data.manifests import read_manifest, write_manifest

log = logging.getLogger("homewav2vec2.transform")


def _filter_valid(manifest_path: Path, invalid_path: Path) -> pd.DataFrame:
    """Return manifest rows minus invalid entries."""
    df = read_manifest(manifest_path)
    if invalid_path.exists():
        inv = read_manifest(invalid_path)
        if not inv.empty and "wav_path" in inv.columns:
            # Only exclude truly unreadable / corrupt (not just warnings)
            hard_issues = inv[inv["reason"].str.contains("unreadable|too_short", na=False)]
            if not hard_issues.empty:
                df = df[~df["wav_path"].isin(hard_issues["wav_path"])].reset_index(drop=True)
    return df


def split_by_column(
    df: pd.DataFrame,
    column: str = "participant_id",
    dev_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group-level split: entire participants go to either train or dev."""
    unique_vals = sorted(df[column].unique())
    rng = pd.np if hasattr(pd, "np") else __import__("numpy").random  # type: ignore[attr-defined]
    import numpy as np

    gen = np.random.RandomState(seed)
    gen.shuffle(unique_vals)  # type: ignore[arg-type]

    n_dev = max(1, int(len(unique_vals) * dev_fraction))
    dev_vals = set(unique_vals[:n_dev])
    train_vals = set(unique_vals[n_dev:])

    dev_df = df[df[column].isin(dev_vals)].reset_index(drop=True)
    train_df = df[df[column].isin(train_vals)].reset_index(drop=True)
    return train_df, dev_df


def run_transformation(cfg: dict) -> None:
    """Entry point called by CLI."""
    artifacts_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    manifest_path = artifacts_dir / "manifests" / "raw_audio_manifest.parquet"
    invalid_path = artifacts_dir / "validation" / "invalid_files.parquet"

    split_col = cfg.get("split_by", "participant_id")
    dev_frac = cfg.get("dev_fraction", 0.1)
    split_seed = cfg.get("split_seed", 42)

    df = _filter_valid(manifest_path, invalid_path)
    log.info("Valid entries after filtering: %d", len(df))

    if df.empty:
        log.warning("No valid entries — cannot create splits.")
        return

    train_df, dev_df = split_by_column(df, column=split_col, dev_fraction=dev_frac, seed=split_seed)

    # Write split manifests
    train_path = artifacts_dir / "manifests" / "train_split.parquet"
    dev_path = artifacts_dir / "manifests" / "dev_split.parquet"
    write_manifest(train_df, train_path)
    write_manifest(dev_df, dev_path)

    # Summary
    summary = {
        "total_valid": len(df),
        "train_files": len(train_df),
        "dev_files": len(dev_df),
        "train_duration_hours": round(train_df["duration_sec"].sum() / 3600, 2),
        "dev_duration_hours": round(dev_df["duration_sec"].sum() / 3600, 2),
        "split_column": split_col,
        "train_groups": sorted(train_df[split_col].unique().tolist()),
        "dev_groups": sorted(dev_df[split_col].unique().tolist()),
    }
    stats_dir = artifacts_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    summary_path = stats_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    log.info("Splits: train=%d  dev=%d  → %s", len(train_df), len(dev_df), summary_path)
