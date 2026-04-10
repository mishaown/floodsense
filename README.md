# FloodSense

> **Notice:** The research paper has been submitted to IEEE Transactions on Geoscience and Remote Sensing.

Lightweight bi-temporal flood detection from SAR satellite imagery using deep learning.

## Features

- **Bi-temporal change detection**: Processes pre/post flood image pairs
- **Modular architecture**: TFEN → parallel CTAM + STSM → MSDAM → PUD with configurable temporal modes
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
python evaluate.py --config configs/s1gfloods.yaml --checkpoint weights/s1gfloods/floodsense/best.pth
python evaluate.py --config configs/sen1floods11.yaml --checkpoint weights/sen1floods11/floodsense/best.pth
python evaluate.py --config configs/ombrias1.yaml --checkpoint weights/ombrias1/floodsense/best.pth
```

### Inference

```bash
# Random samples from dataset
python inference.py --config configs/s1gfloods.yaml --checkpoint weights/s1gfloods/floodsense/best.pth --random-samples 5

# Single image pair
python inference.py --checkpoint weights/s1gfloods/floodsense/best.pth --pre pre_flood.png --post post_flood.png
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
│   └── losses.py         # Loss functions
├── data/
│   ├── sen1floods11.py   # Sen1Floods11 dataset loader
│   ├── s1gfloods.py      # S1GFloods dataset loader
│   ├── ombrias1.py       # OmbriaS1 dataset loader
│   └── preprocessing.py  # SAR preprocessing utilities
└── configs/              # YAML configuration files
    ├── s1gfloods.yaml
    ├── sen1floods11.yaml
    └── ombrias1.yaml
```

## Environment Setup

Create a `.env.local` file in the project root (not committed — machine-specific):

```ini
S1GFLOODS_ROOT=/path/to/S1GFloods
SEN1FLOODS11_ROOT=/path/to/Sen1Floods11
OMBRIAS1_ROOT=/path/to/OmbriaS1

# Optional: override config values
BATCH_SIZE=4
NUM_WORKERS=4
EPOCHS=50
OUTPUT_DIR=outputs
```

## Configuration

Dataset roots are configured in the YAML files under `dataset.root`. Copy and edit one of the provided configs:

```yaml
dataset:
  name: S1GFloods
  root: /path/to/S1GFloods
  image_size: 256
  num_classes: 2
  in_channels: 3
```

## Model Architecture

FloodSense processes bi-temporal image pairs through a modular pipeline:

```
temporal_mode="parallel" (default):
  Pre/Post → TFEN (4 scales) ──► CTAM (per scale) ──────────────► (+) → MSDAM → PUD → Flood Map
                              └──► STSM (per scale, raw feats) ──► (+)          ↓
                                                                             HFFM → Magnitude Map

temporal_mode="all":
  Pre/Post → TFEN (4 scales) → CTAM → STSM → MSDAM → PUD → Flood Map
                                                    ↓
                                                 HFFM → Magnitude Map

temporal_mode="deepest":
  Pre/Post → TFEN → [Scales 1-3: direct] ──────────────→ MSDAM → PUD → Flood Map
                  → [Scale 4: CTAM → STSM] ────────────↗        ↓
                                                             HFFM → Magnitude Map
```

### Components

| Module | Name | Description |
|--------|------|-------------|
| **TFEN** | Temporal Feature Extraction Network | Siamese encoder (EfficientNetV2/ResNet) extracting 4-scale feature pyramids |
| **CTAM** | Cross-Temporal Attention Module | Cross-temporal spatial attention with channel difference weighting |
| **STSM** | Spatial-Temporal Sequence Module | ConvLSTM with hidden/cell state propagation for temporal modeling |
| **MSDAM** | Multi-Scale Difference Aggregation Module | Dual-path (concatenation + absolute difference) feature aggregation |
| **PUD** | Progressive Upsampling Decoder | FPN-style decoder with additive skip connections |
| **HFFM** | Hierarchical Feature Fusion Module | Multi-scale feature aggregation for magnitude estimation |

### Temporal Modes

| Mode | Behavior | When to use |
|------|----------|-------------|
| `parallel` (default) | CTAM and STSM independently process raw encoder features; outputs summed before MSDAM. Direct gradient flow to both modules. | Best for short training budgets |
| `all` | Sequential CTAM → STSM pipeline on all 4 scales. STSM receives CTAM-refined features. | Best with 100+ epochs |
| `deepest` | CTAM + STSM only on the deepest scale; shallow scales go directly to MSDAM. | Resource-constrained settings |

### Usage

```python
from src.model import FloodSenseModelConfig, build_model

# Parallel CTAM+STSM (default)
config = FloodSenseModelConfig(
    encoder="efficientnetv2_rw_t",
    temporal_mode="parallel",
    use_attention=True
)

# Sequential CTAM→STSM (all scales)
config = FloodSenseModelConfig(
    encoder="efficientnetv2_rw_t",
    temporal_mode="all",
    use_attention=True
)

# Efficient mode (deepest layer only)
config = FloodSenseModelConfig(
    encoder="resnet18",
    temporal_mode="deepest",
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
