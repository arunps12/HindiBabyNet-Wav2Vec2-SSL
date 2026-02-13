"""On-the-fly random cropping with optional silence rejection."""

from __future__ import annotations

import numpy as np

from homewav2vec2.dataset.audio_io import get_total_frames, read_audio_window


def compute_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio**2)))


def random_crop(
    wav_path: str,
    crop_sec: float = 8.0,
    target_sr: int = 16000,
    silence_rms_threshold: float = 0.001,
    max_retries: int = 10,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Return a random crop of *crop_sec* seconds from *wav_path*.

    Uses partial reads — never loads the full file.
    Re-samples up to *max_retries* times if crop is below *silence_rms_threshold*.
    """
    if rng is None:
        rng = np.random.RandomState()

    crop_frames = int(crop_sec * target_sr)
    total_frames = get_total_frames(wav_path, target_sr)

    if total_frames <= crop_frames:
        # File shorter than crop — read entire file and pad
        audio = read_audio_window(wav_path, 0, crop_frames, target_sr)
        return audio

    for _ in range(max_retries):
        start = rng.randint(0, total_frames - crop_frames)
        audio = read_audio_window(wav_path, start, crop_frames, target_sr)
        if compute_rms(audio) >= silence_rms_threshold:
            return audio

    # Accept last crop even if silent
    return audio  # type: ignore[possibly-undefined]
