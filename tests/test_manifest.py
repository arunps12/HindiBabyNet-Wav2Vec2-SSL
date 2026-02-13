"""Tests for manifest generation — determinism and correctness."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from homewav2vec2.data.ingest import build_manifest, discover_audio_files
from homewav2vec2.data.manifests import derive_participant_id


def _make_wav(path: Path, duration_sec: float = 5.0, sr: int = 16000) -> None:
    """Create a tiny WAV file for testing."""
    samples = np.random.randn(int(sr * duration_sec)).astype(np.float32) * 0.1
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sr)


def test_discover_finds_wav(tmp_path: Path) -> None:
    _make_wav(tmp_path / "P01" / "a.wav")
    _make_wav(tmp_path / "P01" / "b.wav")
    _make_wav(tmp_path / "P02" / "c.flac", sr=16000)
    files = discover_audio_files(tmp_path)
    assert len(files) == 3


def test_manifest_deterministic(tmp_path: Path) -> None:
    """Running build_manifest twice must produce identical DataFrames."""
    _make_wav(tmp_path / "P01" / "a.wav")
    _make_wav(tmp_path / "P02" / "b.wav")
    _make_wav(tmp_path / "P01" / "c.wav", duration_sec=10)

    df1 = build_manifest(tmp_path, do_sha1=False)
    df2 = build_manifest(tmp_path, do_sha1=False)

    pd.testing.assert_frame_equal(df1, df2)


def test_participant_id_from_folder(tmp_path: Path) -> None:
    p = tmp_path / "SUBJ_42" / "rec.wav"
    _make_wav(p)
    pid = derive_participant_id(p)
    assert pid == "SUBJ_42"
