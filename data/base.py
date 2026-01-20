"""
Base dataset class for FloodSense.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from PIL import Image
import random

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


class BaseFloodDataset(Dataset):
    """Base class for flood detection datasets."""

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 256,
        normalize: bool = True,
        add_ratio: bool = False
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.normalize = normalize
        self.add_ratio = add_ratio
        self.samples: List[Dict] = []
        self.indices: List[int] = []

    def _find_samples(self) -> List[Dict]:
        """Find all samples. To be implemented by subclasses."""
        raise NotImplementedError

    def _split_data(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """Split data into train/val/test."""
        random.seed(42)
        indices = list(range(len(self.samples)))
        random.shuffle(indices)

        n = len(indices)
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)

        if self.split == 'train':
            self.indices = indices[:train_end]
        elif self.split == 'val':
            self.indices = indices[train_end:val_end]
        elif self.split == 'test':
            self.indices = indices[val_end:]
        else:
            self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
