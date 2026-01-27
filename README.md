# FloodSense

Lightweight bi-temporal flood detection from SAR satellite imagery using deep learning.

## Features

- **Bi-temporal change detection**: Processes pre/post flood image pairs
- **Modular architecture**: TFEN → CTAM → STSM → MSDAM → PUD with configurable temporal modes
- **Multiple datasets**: Sen1Floods11, S1GFloods, OmbriaS1
- **Comprehensive metrics**: F1, IoU, Precision, Recall, Kappa, MCC, AUC-ROC

## Installation

```bash
pip install torch torchvision timm tqdm tensorboard pyyaml scikit-learn scikit-image pillow matplotlib
```

## Quick Start

### Training

```bash
python train.py --config configs/s1gfloods.yaml
```

### Evaluation

```bash
python evaluate.py --config configs/s1gfloods.yaml --checkpoint outputs/best.pth
```

### Inference

```bash
# Random samples from dataset
python inference.py --config configs/s1gfloods.yaml --checkpoint outputs/best.pth --random-samples 5

# Single image pair
python inference.py --checkpoint outputs/best.pth --pre pre_flood.png --post post_flood.png
```

## Project Structure

```
FloodSense/
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── inference.py          # Inference script
├── src/
│   ├── model.py          # FloodSense model
│   ├── config.py         # Configuration management
│   ├── metrics.py        # Evaluation metrics
│   ├── losses.py         # Loss functions
│   └── baselines.py      # Baseline models for comparison
├── data/
│   ├── sen1floods11.py   # Sen1Floods11 dataset loader
│   ├── s1gfloods.py      # S1GFloods dataset loader
│   ├── ombrias1.py       # OmbriaS1 dataset loader
│   └── preprocessing.py  # SAR preprocessing utilities
└── configs/              # YAML configuration files
```

## Configuration

Create a `.env.local` file in the project root:

```
S1GFLOODS_ROOT=/path/to/S1GFloods
SEN1FLOODS11_ROOT=/path/to/Sen1Floods11
OMBRIAS1_ROOT=/path/to/OmbriaS1
BATCH_SIZE=16
NUM_WORKERS=4
```

## Model Architecture

FloodSense processes bi-temporal image pairs through a modular pipeline:

```
temporal_mode="all":
  TFEN (4 scales) → CTAM (all) → STSM (all) → MSDAM (all) → PUD → Flood Map
                                                           ↓
                                                         HFFM → CMH → Magnitude Map

temporal_mode="deepest":
  TFEN (4 scales) → [Scales 1-3: direct] ──────────────→ MSDAM → PUD → Flood Map
                  → [Scale 4: CTAM → STSM] ────────────↗       ↓
                                                             HFFM → CMH → Magnitude Map
```

### Components

| Module | Name | Description |
|--------|------|-------------|
| **TFEN** | Temporal Feature Extraction Network | Siamese encoder (EfficientNetV2/ResNet) extracting 4-scale feature pyramids |
| **CTAM** | Cross-Temporal Attention Module | Cross-temporal spatial attention with channel difference weighting |
| **STSM** | Spatial-Temporal Sequence Module | ConvLSTM with proper hidden/cell state propagation for temporal modeling |
| **MSDAM** | Multi-Scale Difference Aggregation Module | Dual-path (concatenation + absolute difference) feature aggregation |
| **PUD** | Progressive Upsampling Decoder | FPN-style decoder with additive skip connections |
| **HFFM** | Hierarchical Feature Fusion Module | Multi-scale feature aggregation for magnitude estimation |
| **CMH** | Change Magnitude Head | Auxiliary head for change intensity prediction |

### Configuration

```python
from src.model import FloodSenseModelConfig, build_model

# Full temporal modeling (all scales)
config = FloodSenseModelConfig(
    encoder="efficientnetv2_rw_t",
    temporal_mode="all",  # CTAM+STSM for all 4 scales
    use_attention=True
)

# Efficient mode (deepest layer only)
config = FloodSenseModelConfig(
    encoder="resnet18",
    temporal_mode="deepest",  # CTAM+STSM only for deepest layer
    use_attention=True
)

model = build_model(config)
```

## Supported Datasets

| Dataset | Type | Input | Resolution |
|---------|------|-------|------------|
| Sen1Floods11 | Sentinel-1 SAR | VV, VH polarization | 256x256 |
| S1GFloods | SAR (8-bit) | Grayscale | 256x256 |
| OmbriaS1 | SAR | Grayscale | 256x256 |

## License

MIT License
