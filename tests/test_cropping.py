"""Tests for on-the-fly cropping — correct length and silence rejection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from homewav2vec2.dataset.cropping import random_crop, compute_rms


def _make_wav(path: Path, duration_sec: float = 30.0, sr: int = 16000, amplitude: float = 0.1) -> None:
    samples = np.random.randn(int(sr * duration_sec)).astype(np.float32) * amplitude
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sr)


def test_crop_correct_length(tmp_path: Path) -> None:
    wav_path = tmp_path / "long.wav"
    _make_wav(wav_path, duration_sec=60.0)

    crop_sec = 8.0
    sr = 16000
    audio = random_crop(str(wav_path), crop_sec=crop_sec, target_sr=sr)

    assert len(audio) == int(crop_sec * sr), f"Expected {int(crop_sec * sr)}, got {len(audio)}"


def test_crop_short_file_padded(tmp_path: Path) -> None:
    """If file is shorter than crop_sec, output should still be crop_sec * sr long."""
    wav_path = tmp_path / "short.wav"
    _make_wav(wav_path, duration_sec=2.0)

    crop_sec = 8.0
    sr = 16000
    audio = random_crop(str(wav_path), crop_sec=crop_sec, target_sr=sr)
    assert len(audio) == int(crop_sec * sr)


def test_silence_rejection_terminates(tmp_path: Path) -> None:
    """Silence rejection must terminate even on fully silent file."""
    wav_path = tmp_path / "silent.wav"
    samples = np.zeros(int(16000 * 30), dtype=np.float32)
    sf.write(str(wav_path), samples, 16000)

    # Should not hang / infinite loop
    audio = random_crop(
        str(wav_path), crop_sec=8.0, target_sr=16000,
        silence_rms_threshold=0.001, max_retries=5,
    )
    assert len(audio) == int(8.0 * 16000)
