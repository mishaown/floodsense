"""
OmbriaS1 Dataset

Sentinel-1 SAR bi-temporal flood detection (grayscale PNG).
Output: 3 channels (triplicated grayscale), labels: 0=no-flood, 255=flood
Note: Histogram matching recommended due to acquisition bias.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Callable, Any
from PIL import Image

from .base import BaseFloodDataset, HAS_ALBUMENTATIONS
from .preprocessing import SARPreprocessor, create_preprocessor_from_config

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    pass


class OmbriaS1Dataset(BaseFloodDataset):
    """
    OmbriaS1 Dataset for SAR bi-temporal flood detection.

    1. Heavy class overlap (Cohen's d = 0.75)
    2. Acquisition bias: Non-flood pixels shift between pre/post images
    3. Minimal flood-specific signal without preprocessing

    RECOMMENDED for OmbriaS1:
    - use_histogram_matching: True (CRITICAL - fixes acquisition bias)
    - use_zscore: True (further standardization)
    - use_log_ratio: True (amplifies change signal)

    Args:
        root: Path to dataset root directory
        split: One of 'train', 'val', 'test', or 'all'
        transform: Albumentations transform pipeline
        image_size: Output image size (default 256)
        normalize: Whether to normalize values to [0, 1]
        add_ratio: Add simple ratio channels (legacy, prefer preprocessing_config)
        preprocessing_config: Dict with preprocessing options:
            - use_histogram_matching: bool (HIGHLY RECOMMENDED)
            - use_zscore: bool (RECOMMENDED)
            - use_adaptive_norm: bool
            - use_log_ratio: bool (RECOMMENDED)
            - use_ndi: bool
            - use_clahe: bool
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 256,
        normalize: bool = True,
        add_ratio: bool = False,
        preprocessing_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(root, split, transform, image_size, normalize, add_ratio)

        # Dataset has predefined train/test splits
        # For val, we'll carve out from train
        if split in ['train', 'val']:
            self.data_dir = self.root / 'train'
        else:
            self.data_dir = self.root / 'test'

        # Directories
        self.before_dir = self.data_dir / 'BEFORE'
        self.after_dir = self.data_dir / 'AFTER'
        self.mask_dir = self.data_dir / 'MASK'

        # Initialize preprocessor if config provided
        self.preprocessor = None
        if preprocessing_config is not None:
            self.preprocessor = create_preprocessor_from_config(preprocessing_config)

        # Find samples
        self.samples = self._find_samples()

        # Split data (only for train/val, test uses all)
        if split in ['train', 'val']:
            self._split_train_val()
        else:
            self.indices = list(range(len(self.samples)))

    def _find_samples(self) -> List[Dict]:
        """Find all samples based on BEFORE images."""
        samples = []

        for before_path in sorted(self.before_dir.glob('S1_before_*.png')):
            # Extract the ID (e.g., 0001 from S1_before_0001.png)
            name = before_path.stem  # S1_before_0001
            sample_id = name.replace('S1_before_', '')  # 0001

            after_path = self.after_dir / f"S1_after_{sample_id}.png"
            mask_path = self.mask_dir / f"S1_mask_{sample_id}.png"

            if after_path.exists() and mask_path.exists():
                samples.append({
                    'name': sample_id,
                    'before': str(before_path),
                    'after': str(after_path),
                    'mask': str(mask_path)
                })

        return samples

    def _split_train_val(self, val_ratio: float = 0.15):
        """Split train data into train/val sets."""
        import random
        random.seed(42)
        indices = list(range(len(self.samples)))
        random.shuffle(indices)

        n = len(indices)
        val_count = int(val_ratio * n)

        if self.split == 'val':
            self.indices = indices[:val_count]
        else:  # train
            self.indices = indices[val_count:]

    def _load_image(self, path: str) -> np.ndarray:
        """Load grayscale SAR image and convert to single channel."""
        img = np.array(Image.open(path).convert('L'))

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        return img

    def _load_mask(self, path: str) -> np.ndarray:
        """Load mask image."""
        mask = np.array(Image.open(path).convert('L'))

        # Convert 255 -> 1 (flood)
        mask = (mask > 127).astype(np.int64)

        return mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = self.indices[idx]
        sample = self.samples[sample_idx]

        # Load images (grayscale, shape: H, W)
        before = self._load_image(sample['before'])
        after = self._load_image(sample['after'])
        mask = self._load_mask(sample['mask'])

        # Resize if needed
        if before.shape[0] != self.image_size or before.shape[1] != self.image_size:
            before = np.array(Image.fromarray((before * 255).astype(np.uint8)).resize(
                (self.image_size, self.image_size), Image.BILINEAR)) / 255.0
            after = np.array(Image.fromarray((after * 255).astype(np.uint8)).resize(
                (self.image_size, self.image_size), Image.BILINEAR)) / 255.0
            mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(
                (self.image_size, self.image_size), Image.NEAREST))

        # Apply advanced preprocessing if configured (on grayscale images)
        # This is CRITICAL for OmbriaS1 due to acquisition bias
        if self.preprocessor is not None:
            before, after, extra_channels = self.preprocessor(before, after)

            # Create 3-channel by triplicating processed grayscale
            before_3ch = np.stack([before, before, before], axis=0)  # (3, H, W)
            after_3ch = np.stack([after, after, after], axis=0)  # (3, H, W)

            # Append extra channels (these are KEY for OmbriaS1)
            extra_list = []
            if 'log_ratio' in extra_channels:
                extra_list.append(extra_channels['log_ratio'])
            if 'ndi' in extra_channels:
                extra_list.append(extra_channels['ndi'])
            if 'difference' in extra_channels:
                extra_list.append(extra_channels['difference'])

            if extra_list:
                extra = np.stack(extra_list, axis=0)
                before_3ch = np.concatenate([before_3ch, extra], axis=0)
                after_3ch = np.concatenate([after_3ch, extra], axis=0)
        else:
            # Create 3-channel images by triplicating grayscale
            # This matches the 3-channel format of Sen1Floods11 (VV, VH, VV+VH)
            before_3ch = np.stack([before, before, before], axis=0)  # (3, H, W)
            after_3ch = np.stack([after, after, after], axis=0)  # (3, H, W)

            if self.add_ratio:
                # Legacy: Add ratio channels to emphasize change (3 ratio channels)
                eps = 1e-6
                ratio = (after - before) / (after + before + eps)
                ratio_3ch = np.stack([ratio, ratio, ratio], axis=0)  # (3, H, W)
                before_3ch = np.concatenate([before_3ch, ratio_3ch], axis=0)  # (6, H, W)
                after_3ch = np.concatenate([after_3ch, ratio_3ch], axis=0)  # (6, H, W)

        num_ch = before_3ch.shape[0]

        # Apply transforms
        if self.transform is not None:
            # Stack for joint transformation
            stacked = np.concatenate([before_3ch, after_3ch], axis=0)  # (2*C, H, W)
            stacked = stacked.transpose(1, 2, 0)  # (H, W, 2*C)

            transformed = self.transform(image=stacked, mask=mask)
            stacked = transformed['image']
            mask = transformed['mask']

            if isinstance(stacked, np.ndarray):
                stacked = stacked.transpose(2, 0, 1)
                before_3ch = stacked[:num_ch]
                after_3ch = stacked[num_ch:]
            else:
                before_3ch = stacked[:num_ch]
                after_3ch = stacked[num_ch:]

            # Ensure label is Long tensor (required by CrossEntropyLoss)
            if isinstance(mask, torch.Tensor):
                mask = mask.long()
            else:
                mask = torch.from_numpy(mask).long()
        else:
            before_3ch = torch.from_numpy(before_3ch).float()
            after_3ch = torch.from_numpy(after_3ch).float()
            mask = torch.from_numpy(mask).long()

        return {
            'pre': before_3ch,
            'post': after_3ch,
            'label': mask,
            'name': sample['name']
        }


def get_ombrias1_train_transforms(image_size: int = 256, add_ratio: bool = False):
    """Get training augmentations for OmbriaS1 SAR data."""
    if not HAS_ALBUMENTATIONS:
        return None

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
        ToTensorV2()
    ])


def get_ombrias1_val_transforms(image_size: int = 256, add_ratio: bool = False):
    """Get validation/test transforms for OmbriaS1 SAR data."""
    if not HAS_ALBUMENTATIONS:
        return None

    return A.Compose([
        ToTensorV2()
    ])
