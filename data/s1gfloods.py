"""
S1GFloods Dataset

SAR bi-temporal flood detection (8-bit grayscale PNG).
Output: 3 channels (triplicated grayscale), labels: 0=no-flood, 255=flood
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


class S1GFloodsDataset(BaseFloodDataset):
    """
    S1GFloods Dataset for SAR bi-temporal flood detection.

    EXCELLENT class separability (Cohen's d = 2.35) 
    Recommended for S1GFloods:
    - Most preprocessing options: False
    - use_clahe: Optional (help with local contrast)

    Args:
        root: Path to dataset root directory
        split: One of 'train', 'val', 'test', or 'all'
        transform: Albumentations transform pipeline
        image_size: Output image size (default 256)
        normalize: Whether to normalize values to [0, 1]
        add_ratio: Add simple ratio channels (legacy, prefer preprocessing_config)
        preprocessing_config: Dict with preprocessing options:
            - use_histogram_matching: bool (not recommended)
            - use_zscore: bool (not recommended)
            - use_adaptive_norm: bool (not needed, 8-bit already normalized)
            - use_log_ratio: bool (not needed, signal already strong)
            - use_ndi: bool
            - use_clahe: bool (optional)
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

        # Directories
        self.a_dir = self.root / 'A'
        self.b_dir = self.root / 'B'
        self.label_dir = self.root / 'Label'

        # Initialize preprocessor if config provided
        self.preprocessor = None
        self.use_difference_channel = False
        if preprocessing_config is not None:
            self.preprocessor = create_preprocessor_from_config(preprocessing_config)
            self.use_difference_channel = preprocessing_config.get('use_difference_channel', False)

        # Find samples
        self.samples = self._find_samples()

        # Split data
        self._split_data()

    def _find_samples(self) -> List[Dict]:
        """Find all samples."""
        samples = []

        for img_path in self.a_dir.glob('*.png'):
            name = img_path.stem
            b_path = self.b_dir / f"{name}.png"
            label_path = self.label_dir / f"{name}.png"

            if b_path.exists() and label_path.exists():
                samples.append({
                    'name': name,
                    'a': str(img_path),
                    'b': str(b_path),
                    'label': str(label_path)
                })

        return samples

    def _load_image(self, path: str) -> np.ndarray:
        """Load image and convert to grayscale, normalized to [0, 1]."""
        # Load as grayscale directly
        img = np.array(Image.open(path).convert('L'))
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

        # Load images as grayscale (H, W)
        img_a = self._load_image(sample['a'])
        img_b = self._load_image(sample['b'])
        label = self._load_mask(sample['label'])

        # Resize if needed
        if img_a.shape[0] != self.image_size or img_a.shape[1] != self.image_size:
            img_a = np.array(Image.fromarray((img_a * 255).astype(np.uint8)).resize(
                (self.image_size, self.image_size), Image.BILINEAR)) / 255.0
            img_b = np.array(Image.fromarray((img_b * 255).astype(np.uint8)).resize(
                (self.image_size, self.image_size), Image.BILINEAR)) / 255.0
            label = np.array(Image.fromarray(label.astype(np.uint8)).resize(
                (self.image_size, self.image_size), Image.NEAREST))

        # Apply advanced preprocessing if configured (on grayscale images)
        if self.preprocessor is not None:
            img_a, img_b, extra_channels = self.preprocessor(img_a, img_b)

            # Create 3-channel by triplicating processed grayscale
            img_a_3ch = np.stack([img_a, img_a, img_a], axis=0)  # (3, H, W)
            img_b_3ch = np.stack([img_b, img_b, img_b], axis=0)  # (3, H, W)

            # Append extra channels
            extra_list = []
            if 'log_ratio' in extra_channels:
                extra_list.append(extra_channels['log_ratio'])
            if 'ndi' in extra_channels:
                extra_list.append(extra_channels['ndi'])
            if self.use_difference_channel and 'difference' in extra_channels:
                extra_list.append(extra_channels['difference'])

            if extra_list:
                extra = np.stack(extra_list, axis=0)
                img_a_3ch = np.concatenate([img_a_3ch, extra], axis=0)
                img_b_3ch = np.concatenate([img_b_3ch, extra], axis=0)
        else:
            # Create 3-channel images by triplicating grayscale
            # This matches the 3-channel format of Sen1Floods11 (VV, VH, VV+VH)
            img_a_3ch = np.stack([img_a, img_a, img_a], axis=0)  # (3, H, W)
            img_b_3ch = np.stack([img_b, img_b, img_b], axis=0)  # (3, H, W)

            if self.add_ratio:
                # Legacy: Add ratio channels to emphasize change (3 ratio channels)
                eps = 1e-6
                ratio = (img_b - img_a) / (img_b + img_a + eps)
                ratio_3ch = np.stack([ratio, ratio, ratio], axis=0)  # (3, H, W)
                img_a_3ch = np.concatenate([img_a_3ch, ratio_3ch], axis=0)  # (6, H, W)
                img_b_3ch = np.concatenate([img_b_3ch, ratio_3ch], axis=0)  # (6, H, W)

        num_ch = img_a_3ch.shape[0]

        # Apply transforms
        if self.transform is not None:
            # Stack images for joint augmentation
            stacked = np.concatenate([img_a_3ch, img_b_3ch], axis=0)  # (2*C, H, W)
            stacked = stacked.transpose(1, 2, 0)  # (H, W, 2*C)

            transformed = self.transform(image=stacked, mask=label)
            stacked = transformed['image']
            label = transformed['mask']

            if isinstance(stacked, torch.Tensor):
                img_a_3ch = stacked[:num_ch]
                img_b_3ch = stacked[num_ch:]
            else:
                stacked = stacked.transpose(2, 0, 1)
                img_a_3ch = stacked[:num_ch]
                img_b_3ch = stacked[num_ch:]

            # Ensure label is Long tensor (required by CrossEntropyLoss)
            if isinstance(label, torch.Tensor):
                label = label.long()
            else:
                label = torch.from_numpy(label).long()
        else:
            img_a_3ch = torch.from_numpy(img_a_3ch).float()
            img_b_3ch = torch.from_numpy(img_b_3ch).float()
            label = torch.from_numpy(label).long()

        return {
            'pre': img_a_3ch,
            'post': img_b_3ch,
            'label': label,
            'name': sample['name']
        }


def get_s1gfloods_train_transforms(image_size: int = 256, add_ratio: bool = False):
    """Get training augmentations for S1GFloods SAR data."""
    if not HAS_ALBUMENTATIONS:
        return None

    # SAR-specific augmentations (no color jitter)
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
        ToTensorV2()
    ])


def get_s1gfloods_val_transforms(image_size: int = 256, add_ratio: bool = False):
    """Get validation/test transforms for S1GFloods SAR data."""
    if not HAS_ALBUMENTATIONS:
        return None

    return A.Compose([
        ToTensorV2()
    ])
