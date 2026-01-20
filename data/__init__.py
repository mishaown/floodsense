"""
FloodSense Data Module

Dataset loaders: Sen1Floods11, S1GFloods, OmbriaS1
SAR preprocessing: histogram matching, z-score, log-ratio, NDI, CLAHE
"""

from typing import Optional, Dict, Any
from torch.utils.data import DataLoader

from .base import BaseFloodDataset, HAS_ALBUMENTATIONS
from .preprocessing import (
    SARPreprocessor,
    create_preprocessor_from_config,
    get_preset_preprocessor,
    PRESET_CONFIGS,
)
from .sen1floods11 import (
    Sen1Floods11Dataset,
    get_sen1floods11_train_transforms,
    get_sen1floods11_val_transforms,
)
from .s1gfloods import (
    S1GFloodsDataset,
    get_s1gfloods_train_transforms,
    get_s1gfloods_val_transforms,
)
from .ombrias1 import (
    OmbriaS1Dataset,
    get_ombrias1_train_transforms,
    get_ombrias1_val_transforms,
)


def get_train_transforms(image_size: int = 256, add_ratio: bool = False):
    """Get training augmentations for SAR data."""
    # All datasets are now SAR, use the same transforms
    return get_sen1floods11_train_transforms(image_size, add_ratio)


def get_val_transforms(image_size: int = 256, add_ratio: bool = False):
    """Get validation/test transforms for SAR data."""
    # All datasets are now SAR, use the same transforms
    return get_sen1floods11_val_transforms(image_size, add_ratio)


def build_dataloader(
    dataset_name: str,
    root: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    image_size: int = 256,
    shuffle: bool = None,
    in_channels: Optional[int] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None,
    use_preset_preprocessing: bool = False
) -> DataLoader:
    """
    Build dataloader for specified dataset.

    Args:
        dataset_name: 'sen1floods11', 's1gfloods', or 'ombrias1'
        root: Dataset root directory
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size
        shuffle: Whether to shuffle (default: True for train)
        in_channels: Desired input channels:
            - All datasets: 3 channels default, 6 with ratio (legacy)
        preprocessing_config: Dict with preprocessing options (see SARPreprocessor)
            Example: {'use_histogram_matching': True, 'use_log_ratio': True}
        use_preset_preprocessing: If True, use dataset-specific preset config
            (overrides preprocessing_config)

    Returns:
        DataLoader instance

    Example with preprocessing:
        # Using custom config
        loader = build_dataloader(
            'ombrias1', root, 'train', 16,
            preprocessing_config={'use_histogram_matching': True, 'use_zscore': True}
        )

        # Using preset (recommended)
        loader = build_dataloader(
            'ombrias1', root, 'train', 16,
            use_preset_preprocessing=True
        )
    """
    if shuffle is None:
        shuffle = (split == 'train')

    # Handle preprocessing config
    preproc_config = None
    if use_preset_preprocessing:
        dataset_key = dataset_name.lower()
        if dataset_key in ['s1gfloods', 's1g']:
            dataset_key = 's1gfloods'
        preproc_config = PRESET_CONFIGS.get(dataset_key)
    elif preprocessing_config is not None:
        preproc_config = preprocessing_config

    # Legacy support for add_ratio
    add_ratio = (in_channels == 6) if preproc_config is None else False

    if split == 'train':
        transform = get_train_transforms(image_size, add_ratio)
    else:
        transform = get_val_transforms(image_size, add_ratio)

    if dataset_name.lower() == 'sen1floods11':
        dataset = Sen1Floods11Dataset(
            root=root,
            split=split,
            transform=transform,
            image_size=image_size,
            add_ratio=add_ratio,
            preprocessing_config=preproc_config
        )
    elif dataset_name.lower() in ['s1gfloods', 's1g']:
        dataset = S1GFloodsDataset(
            root=root,
            split=split,
            transform=transform,
            image_size=image_size,
            add_ratio=add_ratio,
            preprocessing_config=preproc_config
        )
    elif dataset_name.lower() == 'ombrias1':
        dataset = OmbriaS1Dataset(
            root=root,
            split=split,
            transform=transform,
            image_size=image_size,
            add_ratio=add_ratio,
            preprocessing_config=preproc_config
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )


__all__ = [
    # Base classes
    'BaseFloodDataset',
    'HAS_ALBUMENTATIONS',
    # Dataset classes
    'Sen1Floods11Dataset',
    'S1GFloodsDataset',
    'OmbriaS1Dataset',
    # Preprocessing
    'SARPreprocessor',
    'create_preprocessor_from_config',
    'get_preset_preprocessor',
    'PRESET_CONFIGS',
    # Transforms
    'get_train_transforms',
    'get_val_transforms',
    'get_sen1floods11_train_transforms',
    'get_sen1floods11_val_transforms',
    'get_s1gfloods_train_transforms',
    'get_s1gfloods_val_transforms',
    'get_ombrias1_train_transforms',
    'get_ombrias1_val_transforms',
    # Data loader factory
    'build_dataloader',
]
