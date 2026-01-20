# FloodSense

Lightweight bi-temporal flood detection from SAR satellite imagery using deep learning.

## Features

- **Bi-temporal change detection**: Processes pre/post flood image pairs
- **Lightweight architecture**: CNN-LSTM with temporal attention (~14M parameters)
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

FloodSense processes bi-temporal image pairs through:

1. **Shared CNN Encoder**: Siamese weight-shared feature extraction
2. **ConvLSTM**: Temporal sequence modeling
3. **Temporal Attention**: Cross-temporal change highlighting
4. **Multi-scale Difference**: Feature-level change computation
5. **Lightweight Decoder**: FPN-style segmentation head

## Supported Datasets

| Dataset | Type | Input | Resolution |
|---------|------|-------|------------|
| Sen1Floods11 | Sentinel-1 SAR | VV, VH polarization | 256x256 |
| S1GFloods | SAR (8-bit) | Grayscale | 256x256 |
| OmbriaS1 | SAR | Grayscale | 256x256 |

## License

MIT License
