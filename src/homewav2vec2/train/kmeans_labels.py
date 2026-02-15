"""K-means pseudo-label generation for HuBERT SSL pretraining.

Extracts MFCC features from a sample of training audio crops, fits a
MiniBatchKMeans model, and saves it for use during training.

Usage:
    python -m homewav2vec2.train.kmeans_labels \\
        --config src/homewav2vec2/config/training_hubert.yaml \\
        --data-config src/homewav2vec2/config/data.yaml
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml
from sklearn.cluster import MiniBatchKMeans

from homewav2vec2.dataset.hf_dataset import AudioCropDataset
from homewav2vec2.utils.seed import set_seed

log = logging.getLogger("homewav2vec2.kmeans")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_mfcc(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    num_mfcc: int = 13,
    use_deltas: bool = True,
) -> np.ndarray:
    """Extract MFCC features (optionally with delta + delta-delta).

    Args:
        waveform: 1-D float tensor of audio samples.
        sample_rate: Sample rate.
        num_mfcc: Number of MFCC coefficients.
        use_deltas: If True, append delta and delta-delta features.

    Returns:
        ndarray of shape ``(num_frames, feat_dim)`` where
        ``feat_dim = num_mfcc * (3 if use_deltas else 1)``.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (1, T)

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=num_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 320, "n_mels": 80},
    )
    mfcc = mfcc_transform(waveform)  # (1, n_mfcc, T')
    mfcc = mfcc.squeeze(0).T  # (T', n_mfcc)

    if use_deltas:
        delta = torchaudio.functional.compute_deltas(mfcc.T.unsqueeze(0))
        delta2 = torchaudio.functional.compute_deltas(delta)
        mfcc = torch.cat(
            [mfcc, delta.squeeze(0).T, delta2.squeeze(0).T], dim=-1
        )  # (T', 3*n_mfcc)

    return mfcc.numpy()


def _get_feat_extract_output_length(input_length: int, conv_kernel: list, conv_stride: list) -> int:
    """Compute HuBERT/wav2vec2 CNN output length from raw waveform length."""
    for kernel_size, stride in zip(conv_kernel, conv_stride):
        input_length = (input_length - kernel_size) // stride + 1
    return input_length


def align_labels_to_model_frames(
    labels: np.ndarray,
    model_frame_count: int,
) -> np.ndarray:
    """Resample label sequence to match model output frame count.

    Uses nearest-neighbour interpolation to map MFCC-rate labels
    to the model's CNN-output frame rate.
    """
    mfcc_frames = len(labels)
    if mfcc_frames == model_frame_count:
        return labels
    indices = np.round(np.linspace(0, mfcc_frames - 1, model_frame_count)).astype(int)
    return labels[indices]


# ---------------------------------------------------------------------------
# K-means fitting
# ---------------------------------------------------------------------------


def fit_kmeans(
    manifest_path: str | Path,
    cfg: dict,
) -> MiniBatchKMeans:
    """Sample audio crops, extract MFCC features, and fit MiniBatchKMeans.

    Returns:
        Fitted MiniBatchKMeans model.
    """
    seed = cfg.get("seed", 42)
    set_seed(seed)

    crop_sec = cfg.get("crop_sec", 8.0)
    target_sr = cfg.get("sample_rate", 16000)
    num_mfcc = cfg.get("num_mfcc", 13)
    use_deltas = cfg.get("use_deltas", True)
    n_clusters = cfg.get("num_clusters", 100)
    n_samples = cfg.get("kmeans_sample_crops", 50000)
    batch_size = cfg.get("kmeans_batch_size", 10000)

    # Build dataset
    dataset = AudioCropDataset(
        manifest_path=manifest_path,
        crop_sec=crop_sec,
        target_sr=target_sr,
        silence_rms_threshold=cfg.get("silence_rms_threshold", 0.001),
        max_crop_retries=cfg.get("max_crop_retries", 10),
        epoch_multiplier=cfg.get("epoch_multiplier", 1),
    )

    # Limit samples to dataset size
    n_samples = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), size=n_samples, replace=False)

    log.info(
        "Extracting MFCC features from %d crops (n_mfcc=%d, deltas=%s) ...",
        n_samples, num_mfcc, use_deltas,
    )

    all_features: list[np.ndarray] = []
    for i, idx in enumerate(indices):
        sample = dataset[int(idx)]
        waveform = sample["input_values"]
        feats = extract_mfcc(waveform, target_sr, num_mfcc, use_deltas)
        all_features.append(feats)
        if (i + 1) % 5000 == 0:
            log.info("  extracted %d / %d crops", i + 1, n_samples)

    feature_matrix = np.concatenate(all_features, axis=0)
    log.info(
        "Feature matrix: %s  (total frames=%d, dim=%d)",
        feature_matrix.shape, feature_matrix.shape[0], feature_matrix.shape[1],
    )

    log.info("Fitting MiniBatchKMeans with %d clusters ...", n_clusters)
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=seed,
        verbose=1,
        max_iter=150,
    )
    kmeans.fit(feature_matrix)
    log.info(
        "K-means done — inertia=%.2f, iterations=%d",
        kmeans.inertia_, kmeans.n_iter_,
    )
    return kmeans


def save_kmeans(kmeans: MiniBatchKMeans, save_path: str | Path) -> None:
    """Persist the fitted k-means model to disk."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(kmeans, f)
    log.info("K-means model saved to %s", save_path)


def load_kmeans(load_path: str | Path) -> MiniBatchKMeans:
    """Load a previously fitted k-means model."""
    with open(load_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit k-means for HuBERT pseudo-labels")
    parser.add_argument("--config", required=True, help="Path to training_hubert.yaml")
    parser.add_argument("--data-config", default=None, help="Path to data.yaml")
    args = parser.parse_args()

    from homewav2vec2.utils.logging import setup_logging
    setup_logging("INFO")

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    if args.data_config:
        with open(args.data_config) as f:
            data_cfg = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in data_cfg.items() if k not in cfg})

    manifest_path = Path(cfg.get("artifacts_dir", "artifacts")) / "manifests" / "train_split.parquet"
    kmeans_dir = Path(cfg.get("kmeans_dir", "artifacts/hubert_training/kmeans"))

    kmeans = fit_kmeans(manifest_path, cfg)
    save_kmeans(kmeans, kmeans_dir / "kmeans_model.pkl")


if __name__ == "__main__":
    main()
