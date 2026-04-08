"""
Dataset Loaders for FloodSense.

This module re-exports all dataset classes and utilities from the data package
for backward compatibility.

Supports:
- Sen1Floods11: SAR bi-temporal flood detection (3 channels: VV, VH, VV+VH)
- S1GFloods: SAR bi-temporal flood detection (3 channels: triplicated grayscale)
- OmbriaS1: SAR bi-temporal flood detection (3 channels: triplicated grayscale)

All datasets output 3 channels by default, or 6 channels with ratio.
"""

from data import (
    BaseFloodDataset,
    Sen1Floods11Dataset,
    S1GFloodsDataset,
    OmbriaS1Dataset,
    get_train_transforms,
    get_val_transforms,
    build_dataloader,
    HAS_ALBUMENTATIONS,
)

__all__ = [
    'BaseFloodDataset',
    'Sen1Floods11Dataset',
    'S1GFloodsDataset',
    'OmbriaS1Dataset',
    'get_train_transforms',
    'get_val_transforms',
    'build_dataloader',
    'HAS_ALBUMENTATIONS',
]
