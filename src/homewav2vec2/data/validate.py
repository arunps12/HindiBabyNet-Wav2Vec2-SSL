"""Phase 2 — Data validation: detect issues without deleting anything."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from homewav2vec2.data.manifests import read_manifest, write_manifest

log = logging.getLogger("homewav2vec2.validate")


def _check_readable(wav_path: str) -> tuple[bool, str]:
    """Try opening the file with soundfile."""
    try:
        sf.info(wav_path)
        return True, ""
    except Exception as exc:
        return False, f"unreadable: {exc}"


def _check_rms(wav_path: str, max_frames: int = 16000 * 30) -> float:
    """Compute RMS of first *max_frames* samples (mono mix)."""
    try:
        data, sr = sf.read(wav_path, frames=max_frames, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        rms = float(np.sqrt(np.mean(mono**2)))
        return rms
    except Exception:
        return -1.0


def validate_manifest(
    manifest_path: str | Path,
    min_duration_sec: float = 2.0,
    silence_rms_threshold: float = 0.001,
    clipping_threshold: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Validate every entry in the manifest.

    Returns:
        valid_df   — rows that pass all checks
        invalid_df — rows with at least one issue (+ ``reason`` column)
        summary    — dict with aggregate stats
    """
    df = read_manifest(manifest_path)
    log.info("Loaded manifest with %d entries", len(df))

    issues: list[dict] = []
    rms_values: list[float] = []

    for idx, row in df.iterrows():
        wav_path = row["wav_path"]
        reasons: list[str] = []

        # 1) Readability
        ok, msg = _check_readable(wav_path)
        if not ok:
            reasons.append(msg)
            issues.append({"index": idx, "wav_path": wav_path, "reason": "; ".join(reasons)})
            rms_values.append(-1.0)
            continue

        # 2) Duration
        if row["duration_sec"] < min_duration_sec:
            reasons.append(f"too_short ({row['duration_sec']:.2f}s < {min_duration_sec}s)")

        # 3) RMS / silence
        rms = _check_rms(wav_path)
        rms_values.append(rms)
        if 0 <= rms < silence_rms_threshold:
            reasons.append(f"near_silent (rms={rms:.6f})")

        # 4) Channels note
        if row.get("channels", 1) > 1:
            reasons.append(f"multi_channel ({row['channels']}ch, will convert to mono)")

        if reasons:
            issues.append({"index": idx, "wav_path": wav_path, "reason": "; ".join(reasons)})

        if (idx + 1) % 50 == 0:  # type: ignore[operator]
            log.info("Validated %d / %d …", idx + 1, len(df))

    df["rms"] = rms_values

    invalid_df = pd.DataFrame(issues)
    # Keep all rows in audio_stats
    stats_df = df[["wav_path", "recording_id", "participant_id", "duration_sec",
                    "sample_rate", "channels", "file_size_bytes", "rms"]].copy()

    summary = {
        "total_files": len(df),
        "invalid_files": len(invalid_df),
        "valid_files": len(df) - len(invalid_df),
        "total_duration_hours": round(df["duration_sec"].sum() / 3600, 2),
        "mean_duration_sec": round(df["duration_sec"].mean(), 2) if len(df) else 0,
    }

    return stats_df, invalid_df, summary


def run_validation(cfg: dict) -> None:
    """Entry point called by CLI."""
    artifacts_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    manifest_path = artifacts_dir / "manifests" / "raw_audio_manifest.parquet"

    min_dur = cfg.get("min_duration_sec", 2.0)
    silence_thr = cfg.get("silence_rms_threshold", 0.001)
    clip_thr = cfg.get("clipping_threshold", 0.99)

    stats_df, invalid_df, summary = validate_manifest(
        manifest_path,
        min_duration_sec=min_dur,
        silence_rms_threshold=silence_thr,
        clipping_threshold=clip_thr,
    )

    # Write outputs
    val_dir = artifacts_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = artifacts_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    write_manifest(invalid_df, val_dir / "invalid_files.parquet")
    write_manifest(stats_df, stats_dir / "audio_stats.parquet")

    summary_path = val_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    log.info("Validation complete: %s", summary)
    log.info("Outputs: %s, %s, %s",
             val_dir / "invalid_files.parquet",
             stats_dir / "audio_stats.parquet",
             summary_path)
