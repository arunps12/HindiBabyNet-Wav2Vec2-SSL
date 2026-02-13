"""Manifest I/O helpers — create, read, and write manifests as Parquet."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import soundfile as sf


def compute_sha1(path: Path) -> str:
    """Compute SHA-1 of a file (streaming)."""
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_audio(path: Path) -> dict:
    """Return metadata dict for a single audio file (no full load)."""
    info = sf.info(str(path))
    return {
        "wav_path": str(path),
        "recording_id": path.stem,
        "duration_sec": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "file_size_bytes": path.stat().st_size,
    }


def derive_participant_id(path: Path) -> str:
    """Best-effort extraction of participant ID from folder / filename.

    Convention: the parent directory name is the participant ID, e.g.
      .../RawAudioData/<participant_id>/<file>.wav
    Falls back to 'unknown'.
    """
    parent = path.parent.name
    return parent if parent and parent != "RawAudioData" else "unknown"


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported manifest format: {path.suffix}")


def write_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported manifest format: {path.suffix}")
    return path
