"""Tests for HuBERT pretraining — config parsing, k-means, model wrapper."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml


def _make_wav(path: Path, duration_sec: float = 30.0, sr: int = 16000) -> None:
    samples = np.random.randn(int(sr * duration_sec)).astype(np.float32) * 0.1
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sr)


def test_hubert_training_config_parses() -> None:
    """Verify training_hubert.yaml can be loaded."""
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "src" / "homewav2vec2" / "config" / "training_hubert.yaml"
    )
    if not cfg_path.exists():
        return

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    assert cfg["base_model"] == "facebook/hubert-base-ls960"
    assert cfg["fp16"] is True
    assert cfg["gradient_checkpointing"] is True
    assert cfg["num_clusters"] == 100
    assert cfg["kmeans_feature"] == "mfcc"
    assert cfg["num_mfcc"] == 13
    assert cfg["per_gpu_batch"] == 2


def test_mfcc_extraction() -> None:
    """Verify MFCC extraction produces correct shape."""
    import torch
    from homewav2vec2.train.kmeans_labels import extract_mfcc

    # 2 seconds of audio at 16kHz
    waveform = torch.randn(32000)
    feats = extract_mfcc(waveform, sample_rate=16000, num_mfcc=13, use_deltas=True)

    assert feats.ndim == 2
    assert feats.shape[1] == 39  # 13 * 3 (mfcc + delta + delta-delta)
    assert feats.shape[0] > 0  # some frames

    feats_no_delta = extract_mfcc(waveform, sample_rate=16000, num_mfcc=13, use_deltas=False)
    assert feats_no_delta.shape[1] == 13


def test_label_alignment() -> None:
    """Verify label alignment to model frame count."""
    from homewav2vec2.train.kmeans_labels import align_labels_to_model_frames

    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Same length
    aligned = align_labels_to_model_frames(labels, 10)
    assert len(aligned) == 10
    np.testing.assert_array_equal(aligned, labels)

    # Downsample
    aligned = align_labels_to_model_frames(labels, 5)
    assert len(aligned) == 5

    # Upsample
    aligned = align_labels_to_model_frames(labels, 20)
    assert len(aligned) == 20


def test_hubert_model_wrapper_forward() -> None:
    """Smoke test: HubertForPreTraining forward pass on tiny random data."""
    import torch
    try:
        from transformers import HubertConfig, HubertModel
    except ImportError:
        return  # skip if transformers not available

    from homewav2vec2.train.run_pretrain_hubert import HubertForPreTraining

    # Create a tiny HuBERT config for testing (no download)
    config = HubertConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        conv_dim=[32, 32, 32, 32, 32, 32, 32],
        conv_kernel=[10, 3, 3, 3, 3, 2, 2],
        conv_stride=[5, 2, 2, 2, 2, 2, 2],
        num_feat_extract_layers=7,
    )
    hubert_base = HubertModel(config)
    model = HubertForPreTraining(hubert_base, num_clusters=50, final_proj_dim=16)

    # Create a batch: 2 samples, ~1.6 seconds at 16kHz = 25600 samples
    input_values = torch.randn(2, 25600)

    # Compute expected frame count
    from homewav2vec2.train.run_pretrain_hubert import _feat_extract_output_lengths
    seq_len = _feat_extract_output_lengths(25600, config)

    mask = torch.zeros(2, seq_len, dtype=torch.bool)
    mask[:, 5:15] = True  # mask 10 frames
    labels = torch.randint(0, 50, (2, seq_len))

    output = model(input_values, mask_time_indices=mask, labels=labels)

    assert output.loss is not None
    assert output.loss.item() > 0
    assert output.logits.shape == (2, seq_len, 50)


def test_kmeans_fit_smoke(tmp_path: Path) -> None:
    """Smoke test: k-means fitting on tiny data."""
    from sklearn.cluster import MiniBatchKMeans

    # Simulate MFCC features (100 frames, 39-dim)
    features = np.random.randn(100, 39).astype(np.float32)
    kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, max_iter=10)
    kmeans.fit(features)

    labels = kmeans.predict(features[:10])
    assert labels.shape == (10,)
    assert all(0 <= l < 5 for l in labels)

    # Test save/load
    from homewav2vec2.train.kmeans_labels import save_kmeans, load_kmeans

    save_path = tmp_path / "kmeans.pkl"
    save_kmeans(kmeans, save_path)
    loaded = load_kmeans(save_path)
    np.testing.assert_array_equal(loaded.predict(features[:10]), labels)
