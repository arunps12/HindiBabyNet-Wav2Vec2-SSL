"""Tests for validation — flags known bad inputs correctly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from homewav2vec2.data.ingest import build_manifest
from homewav2vec2.data.manifests import write_manifest
from homewav2vec2.data.validate import validate_manifest


def _make_wav(path: Path, duration_sec: float = 5.0, sr: int = 16000, amplitude: float = 0.1) -> None:
    samples = np.random.randn(int(sr * duration_sec)).astype(np.float32) * amplitude
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sr)


def _make_silent_wav(path: Path, duration_sec: float = 5.0, sr: int = 16000) -> None:
    samples = np.zeros(int(sr * duration_sec), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sr)


def test_validates_short_file(tmp_path: Path) -> None:
    _make_wav(tmp_path / "P01" / "short.wav", duration_sec=0.5)
    _make_wav(tmp_path / "P01" / "good.wav", duration_sec=10.0)

    df = build_manifest(tmp_path)
    mf_path = tmp_path / "manifest.parquet"
    write_manifest(df, mf_path)

    _, invalid, summary = validate_manifest(mf_path, min_duration_sec=2.0)
    assert summary["invalid_files"] >= 1
    assert any("too_short" in str(r) for r in invalid["reason"].tolist())


def test_validates_silent_file(tmp_path: Path) -> None:
    _make_silent_wav(tmp_path / "P01" / "silent.wav", duration_sec=5.0)
    _make_wav(tmp_path / "P01" / "good.wav", duration_sec=10.0)

    df = build_manifest(tmp_path)
    mf_path = tmp_path / "manifest.parquet"
    write_manifest(df, mf_path)

    _, invalid, _ = validate_manifest(mf_path, silence_rms_threshold=0.001)
    assert any("near_silent" in str(r) for r in invalid["reason"].tolist())
