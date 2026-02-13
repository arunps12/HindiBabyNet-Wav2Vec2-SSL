"""Phase 1 — Data ingestion: build a deterministic manifest of raw audio."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from homewav2vec2.data.manifests import (
    compute_sha1,
    derive_participant_id,
    probe_audio,
    write_manifest,
)

log = logging.getLogger("homewav2vec2.ingest")


def discover_audio_files(
    root: str | Path,
    allowed_ext: list[str] | None = None,
) -> list[Path]:
    """Recursively find audio files under *root*, sorted for determinism."""
    root = Path(root)
    if allowed_ext is None:
        allowed_ext = [".wav", ".flac"]
    files = [
        p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in allowed_ext
    ]
    return files


def build_manifest(
    raw_audio_root: str | Path,
    allowed_ext: list[str] | None = None,
    do_sha1: bool = False,
) -> pd.DataFrame:
    """Scan *raw_audio_root* and return a manifest DataFrame."""
    audio_files = discover_audio_files(raw_audio_root, allowed_ext)
    log.info("Found %d audio files under %s", len(audio_files), raw_audio_root)

    rows: list[dict] = []
    for i, path in enumerate(audio_files):
        try:
            meta = probe_audio(path)
            meta["participant_id"] = derive_participant_id(path)
            if do_sha1:
                meta["sha1"] = compute_sha1(path)
            rows.append(meta)
        except Exception as exc:
            log.warning("Skipping %s — probe failed: %s", path, exc)
        if (i + 1) % 50 == 0:
            log.info("Probed %d / %d files …", i + 1, len(audio_files))

    df = pd.DataFrame(rows)
    # Deterministic ordering
    if not df.empty:
        df = df.sort_values(["participant_id", "recording_id"]).reset_index(drop=True)
    return df


def run_ingestion(cfg: dict) -> Path:
    """Entry point called by CLI — reads config dict, writes manifest."""
    raw_audio_root = cfg["raw_audio_root"]
    allowed_ext = cfg.get("allowed_ext", [".wav", ".flac"])
    do_sha1 = cfg.get("compute_sha1", False)
    artifacts_dir = Path(cfg.get("artifacts_dir", "artifacts"))

    manifest_path = artifacts_dir / "manifests" / "raw_audio_manifest.parquet"

    df = build_manifest(raw_audio_root, allowed_ext=allowed_ext, do_sha1=do_sha1)
    out = write_manifest(df, manifest_path)
    log.info("Manifest written to %s  (%d rows)", out, len(df))
    return out
