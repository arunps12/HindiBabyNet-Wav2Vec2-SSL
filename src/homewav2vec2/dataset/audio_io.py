"""Low-level audio I/O — partial reads, resampling, mono conversion."""

from __future__ import annotations

import numpy as np
import soundfile as sf


def read_audio_window(
    path: str,
    start_frame: int,
    num_frames: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """Read a window of audio from *path* WITHOUT loading the full file.

    Returns mono float32 numpy array at *target_sr*.
    Resampling is done only when the source sample rate differs.
    """
    info = sf.info(path)
    src_sr = info.samplerate

    # Adjust frame indices if we need to resample
    if src_sr != target_sr:
        ratio = src_sr / target_sr
        src_start = int(start_frame * ratio)
        src_frames = int(num_frames * ratio)
    else:
        src_start = start_frame
        src_frames = num_frames

    # Clamp to file length
    total_frames = info.frames
    src_start = min(src_start, max(0, total_frames - src_frames))
    src_start = max(0, src_start)
    src_frames = min(src_frames, total_frames - src_start)

    data, file_sr = sf.read(
        path, start=src_start, frames=src_frames, dtype="float32", always_2d=True,
    )

    # Convert to mono
    mono = data.mean(axis=1)

    # Resample if needed (simple linear interpolation — acceptable for SSL pretraining)
    if file_sr != target_sr:
        import torchaudio
        import torch

        waveform = torch.from_numpy(mono).unsqueeze(0)  # (1, T)
        resampled = torchaudio.functional.resample(waveform, file_sr, target_sr)
        mono = resampled.squeeze(0).numpy()

    # Ensure exactly num_frames
    if len(mono) > num_frames:
        mono = mono[:num_frames]
    elif len(mono) < num_frames:
        mono = np.pad(mono, (0, num_frames - len(mono)), mode="constant")

    return mono


def get_total_frames(path: str, target_sr: int = 16000) -> int:
    """Return the number of frames at *target_sr* (without loading audio)."""
    info = sf.info(path)
    if info.samplerate == target_sr:
        return info.frames
    return int(info.frames * target_sr / info.samplerate)
