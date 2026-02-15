---
language:
  - hi
license: apache-2.0
tags:
  - wav2vec2
  - self-supervised
  - speech
  - audio
  - hindi
  - infant
  - child-directed-speech
  - home-recordings
  - ssl
  - pretraining
datasets:
  - custom
library_name: transformers
pipeline_tag: feature-extraction
base_model: facebook/wav2vec2-base
model-index:
  - name: wav2vec2-home-hindibabynet-ssl
    results: []
---

# Wav2Vec2 Home-Domain SSL — HindiBabyNet

**Self-supervised continued pretraining** of [`facebook/wav2vec2-base`](https://huggingface.co/facebook/wav2vec2-base) on naturalistic Hindi parent–infant home interaction recordings from the HindiBabyNet corpus.

## Model Description

This model adapts the wav2vec2-base speech representation to the **home-recording domain** — noisy, reverberant, multi-speaker environments with infant vocalisations and Hindi child-directed speech (CDS). The goal is to learn robust latent audio representations that better capture the acoustic characteristics of naturalistic home environments, which differ substantially from the read/broadcast speech used to train the original wav2vec2-base.

The model was pretrained using the standard wav2vec2 **contrastive + diversity** self-supervised objective (mask-and-predict on quantised latent speech frames), without any transcription labels.

| Property | Value |
|---|---|
| **Base model** | `facebook/wav2vec2-base` |
| **Architecture** | `Wav2Vec2ForPreTraining` |
| **Parameters** | ~95 M |
| **Hidden size** | 768 |
| **Attention heads** | 12 |
| **Transformer layers** | 12 |
| **Feature extractor** | 7-layer CNN |
| **Quantiser** | 2 groups × 320 codebook entries |

## Training Data

The model was trained on **~346 hours** of naturalistic home recordings from the HindiBabyNet corpus — a collection of day-long audio recordings of Hindi-speaking parent–infant dyads in their home environment.

| Statistic | Value |
|---|---|
| **Total audio** | ~346 hours |
| **Number of recordings** | 111 files |
| **Train split** | 99 files (~90%) |
| **Dev split** | 12 files (~10%) |
| **Split strategy** | by participant ID (no leakage) |
| **Language** | Hindi |
| **Domain** | Home, parent–infant interaction |

**Audio characteristics:**
- Long-form naturalistic recordings (minutes to hours per file)
- Multi-speaker: adults, infants, children, background voices
- Real-world noise: TV, kitchen, traffic, household appliances
- Reverberant home acoustics
- Child-directed speech, infant vocalisations, babbling

## Training Procedure

### Preprocessing

- Resampled to **16 kHz mono** on-the-fly
- **Random cropping**: 8-second crops drawn randomly from long recordings (no chunk files saved to disk)
- Silence rejection: crops below RMS threshold 0.001 are re-sampled (up to 10 retries)
- Each file yields **10 random crops per epoch** (`epoch_multiplier=10`)

### Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| **Optimiser** | AdamW |
| **Learning rate** | 5e-5 |
| **LR scheduler** | Linear with warmup |
| **Warmup steps** | 5,000 |
| **Total training steps** | 50,000 |
| **Effective batch size** | 64 (2 per GPU × 4 GPUs × 8 grad accumulation) |
| **Precision** | FP16 (mixed precision) |
| **Gradient checkpointing** | Enabled |
| **Max gradient norm** | 1.0 |
| **Weight decay** | 0.01 |
| **Crop duration** | 8.0 seconds |
| **Mask time prob** | 0.05 |
| **Mask time length** | 10 frames |
| **Num negatives** | 100 |
| **Contrastive temperature** | 0.1 |
| **Diversity loss weight** | 0.1 |

### Training Infrastructure

- **Hardware**: 4 × NVIDIA RTX 2080 Ti (11 GB VRAM each)
- **Distributed**: PyTorch DDP via `torchrun`
- **Software**: PyTorch 2.10, Transformers 5.1.0, torchaudio 2.10
- **Training time**: ~50,000 steps over ~3,333 effective epochs

### Training Loss

| Metric | Value |
|---|---|
| **Initial loss** (step 100) | 5,191 |
| **Final loss** (step 50,000) | 1,662 |
| **Avg loss** (first 1k steps) | 4,740 |
| **Avg loss** (last 1k steps) | 2,815 |

The contrastive + diversity loss decreased consistently over training, indicating successful domain adaptation of the speech representations.

## Intended Use

### Primary Use Case

This model is intended as a **feature extractor / encoder** for downstream speech tasks on Hindi home-domain audio, such as:

- Automatic speech recognition (ASR) of child-directed speech
- Speaker diarisation in home recordings
- Infant vocalisation detection and classification
- Child language development research
- Acoustic event detection in home environments

### How to Use

```python
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import torch
import torchaudio

# Load model and feature extractor
model = Wav2Vec2Model.from_pretrained("arunps/wav2vec2-home-hindibabynet-ssl")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("arunps/wav2vec2-home-hindibabynet-ssl")

# Load and preprocess audio
waveform, sr = torchaudio.load("your_audio.wav")
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
waveform = waveform.mean(dim=0)  # mono

# Extract features
inputs = feature_extractor(waveform.numpy(), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Last hidden state: (batch, time_frames, 768)
hidden_states = outputs.last_hidden_state
```

### Fine-tuning for ASR

```python
from transformers import Wav2Vec2ForCTC

model = Wav2Vec2ForCTC.from_pretrained(
    "arunps/wav2vec2-home-hindibabynet-ssl",
    ctc_loss_reduction="mean",
    pad_token_id=0,
    vocab_size=YOUR_VOCAB_SIZE,  # set to your tokenizer vocab
)
# Freeze feature extractor, fine-tune transformer + CTC head
model.freeze_feature_encoder()
```

## Limitations

- **Domain-specific**: Optimised for Hindi home recordings; may not generalise well to studio/broadcast audio.
- **No labels used**: This is a self-supervised model — it has not been fine-tuned on any labelled task. Downstream fine-tuning is required for ASR, classification, etc.
- **Language**: Trained exclusively on Hindi home audio; cross-lingual transfer has not been evaluated.
- **Noise**: While the model is trained on noisy home audio (which may improve robustness), extreme noise conditions were not filtered out.

## Ethical Considerations

- The training data consists of naturalistic home recordings of families with young children. All data collection was conducted under appropriate ethical review and informed consent.
- The model does not perform speech recognition or speaker identification on its own — it produces general-purpose speech representations.
- Users should ensure compliance with applicable data protection regulations when applying this model to new audio data.

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{wav2vec2-home-hindibabynet-ssl,
  author       = {Arun P S},
  title        = {Wav2Vec2 Home-Domain SSL for HindiBabyNet},
  year         = {2026},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/arunps/wav2vec2-home-hindibabynet-ssl}
}
```

## Model Card Contact

**Arun P S** — [Hugging Face profile](https://huggingface.co/arunps)
