"""
Sen1Floods11 Dataset

Sentinel-1 SAR bi-temporal flood detection.
Output: 3 channels (VV, VH, VV+VH), labels: -1=ignore, 0=no-flood, 1=flood
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Callable, Any

from .base import BaseFloodDataset, tifffile, HAS_ALBUMENTATIONS
from .preprocessing import SARPreprocessor, create_preprocessor_from_config

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    pass


class Sen1Floods11Dataset(BaseFloodDataset):
    """
    Sen1Floods11 Change Detection Dataset for SAR imagery.

    Recommended for Sen1Floods11:
    - use_adaptive_norm: True (handles varying SAR conditions)
    - use_log_ratio: True (amplifies subtle flood signal)

    Args:
        root: Path to dataset root directory
        split: One of 'train', 'val', 'test', or 'all'
        transform: Albumentations transform pipeline
        image_size: Output image size (default 256)
        normalize: Whether to normalize SAR values to [0, 1]
        add_ratio: Add simple ratio channels (legacy, prefer preprocessing_config)
        preprocessing_config: Dict with preprocessing options:
            - use_histogram_matching: bool (not recommended for this dataset)
            - use_zscore: bool
            - use_adaptive_norm: bool (recommended)
            - use_log_ratio: bool
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

        # Directories
        self.pre_dir = self.root / 'PRE_S1'
        self.post_dir = self.root / 'POST_S1'
        self.label_dir = self.root / 'Labels'

        # Initialize preprocessor if config provided
        self.preprocessor = None
        self.use_difference_channel = False
        if preprocessing_config is not None:
            self.preprocessor = create_preprocessor_from_config(preprocessing_config)
            self.use_difference_channel = preprocessing_config.get('use_difference_channel', False)

        # Find matching samples
        self.samples = self._find_samples()

        # Split data
        self._split_data()

    def _find_samples(self) -> List[Dict]:
        """Find matching pre/post/label triplets."""
        samples = []

        # Get all label files
        label_files = list(self.label_dir.glob('*.tif'))

        for label_path in label_files:
            # Parse name: {Country}_{ID}_LabelHand.tif
            name = label_path.stem.replace('_LabelHand', '')

            # Find matching PRE and POST files
            pre_path = self.pre_dir / f"{name}_S1_PREHand.tif"
            post_path = self.post_dir / f"{name}_S1Hand.tif"

            if pre_path.exists() and post_path.exists():
                samples.append({
                    'name': name,
                    'pre': str(pre_path),
                    'post': str(post_path),
                    'label': str(label_path)
                })

        return samples

    def _load_sar(self, path: str) -> np.ndarray:
        """Load SAR image and create 3-channel output (VV, VH, VV+VH)."""
        if tifffile is None:
            raise ImportError("tifffile required for Sen1Floods11")

        img = tifffile.imread(path)

        # Ensure shape is (2, H, W) - VV and VH bands
        if img.ndim == 3 and img.shape[-1] == 2:
            img = img.transpose(2, 0, 1)

        # Handle NaN values
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize SAR values (dB scale typically -30 to 0)
        if self.normalize:
            img = np.clip(img, -30, 0)
            img = (img + 30) / 30  # Scale to [0, 1]

        # Create 3rd channel: VV + VH (normalized sum)
        vv = img[0:1]  # (1, H, W)
        vh = img[1:2]  # (1, H, W)
        vv_vh_sum = (vv + vh) / 2.0  # Average to keep in [0, 1] range

        # Stack to create 3-channel image: (VV, VH, VV+VH)
        img = np.concatenate([vv, vh, vv_vh_sum], axis=0)  # (3, H, W)

        return img.astype(np.float32)

    def _load_label(self, path: str) -> np.ndarray:
        """Load label image."""
        if tifffile is None:
            raise ImportError("tifffile required for Sen1Floods11")

        label = tifffile.imread(path)

        # Convert -1 to ignore index (255)
        label = label.astype(np.int64)
        label[label == -1] = 255

        return label

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = self.indices[idx]
        sample = self.samples[sample_idx]

        # Load data (3 channels: VV, VH, VV+VH)
        pre = self._load_sar(sample['pre'])
        post = self._load_sar(sample['post'])
        label = self._load_label(sample['label'])

        # Resize if needed
        if pre.shape[1] != self.image_size or pre.shape[2] != self.image_size:
            from skimage.transform import resize
            pre = resize(pre, (3, self.image_size, self.image_size), preserve_range=True)
            post = resize(post, (3, self.image_size, self.image_size), preserve_range=True)
            label = resize(label, (self.image_size, self.image_size), order=0, preserve_range=True)
            pre = pre.astype(np.float32)
            post = post.astype(np.float32)
            label = label.astype(np.int64)

        # Apply advanced preprocessing if configured
        if self.preprocessor is not None:
            pre, post, extra_channels = self.preprocessor(pre, post)

            # Append extra channels (log_ratio, ndi, difference, etc.)
            extra_list = []
            if 'log_ratio' in extra_channels:
                extra_list.append(extra_channels['log_ratio'])
            if 'ndi' in extra_channels:
                extra_list.append(extra_channels['ndi'])
            if self.use_difference_channel and 'difference' in extra_channels:
                extra_list.append(extra_channels['difference'])

            if extra_list:
                extra = np.concatenate(extra_list, axis=0) if extra_list[0].ndim == 3 else np.stack(extra_list, axis=0)
                pre = np.concatenate([pre, extra], axis=0)
                post = np.concatenate([post, extra], axis=0)

        elif self.add_ratio:
            # Legacy: Add ratio channels to emphasize change (3 ratio channels for VV, VH, VV+VH)
            eps = 1e-6
            ratio = (post - pre) / (post + pre + eps)
            pre = np.concatenate([pre, ratio], axis=0)  # (6, H, W)
            post = np.concatenate([post, ratio], axis=0)  # (6, H, W)

        num_ch = pre.shape[0]

        # Apply transforms (custom or albumentations)
        if self.transform is not None:
            # Stack for joint transformation
            stacked = np.concatenate([pre, post], axis=0)  # (2*C, H, W)
            stacked = stacked.transpose(1, 2, 0)  # (H, W, 2*C)

            transformed = self.transform(image=stacked, mask=label)
            stacked = transformed['image']
            label = transformed['mask']

            if isinstance(stacked, np.ndarray):
                stacked = stacked.transpose(2, 0, 1)
                pre = stacked[:num_ch]
                post = stacked[num_ch:]
            else:
                pre = stacked[:num_ch]
                post = stacked[num_ch:]

            # Ensure label is Long tensor (required by CrossEntropyLoss)
            if isinstance(label, torch.Tensor):
                label = label.long()
            else:
                label = torch.from_numpy(label).long()
        else:
            pre = torch.from_numpy(pre)
            post = torch.from_numpy(post)
            label = torch.from_numpy(label).long()

        return {
            'pre': pre,
            'post': post,
            'label': label,
            'name': sample['name']
        }


def get_sen1floods11_train_transforms(image_size: int = 256, add_ratio: bool = False):
    """Get training augmentations for SAR data."""
    if not HAS_ALBUMENTATIONS:
        return None

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
        ToTensorV2()
    ])


def get_sen1floods11_val_transforms(image_size: int = 256, add_ratio: bool = False):
    """Get validation/test transforms for SAR data."""
    if not HAS_ALBUMENTATIONS:
        return None

    return A.Compose([
        ToTensorV2()
    ])
