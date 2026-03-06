"""
Configuration management for FloodSense.
Loads from YAML configs with .env.local overrides.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict
from pathlib import Path


@dataclass
class DatasetConfig:
    name: str = "S1GFloods"
    root: str = ""
    image_size: int = 256
    num_classes: int = 2
    in_channels: int = 3


@dataclass
class ModelConfig:
    encoder: str = "resnet18"
    pretrained: bool = True
    hidden_dim: int = 64
    lstm_hidden: int = 128
    lstm_layers: int = 1
    use_attention: bool = True
    dropout: float = 0.1


@dataclass
class PreprocessingConfig:
    use_histogram_matching: bool = False
    use_zscore: bool = False
    use_adaptive_norm: bool = False
    adaptive_percentile_low: float = 2.0
    adaptive_percentile_high: float = 98.0
    use_log_ratio: bool = False
    log_ratio_clip: float = 3.0
    use_ndi: bool = False
    use_clahe: bool = False
    use_difference_channel: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    eps: float = 1e-6


@dataclass
class OptimizerConfig:
    type: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 0.01


@dataclass
class SchedulerConfig:
    type: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    gradient_clip: float = 1.0
    mixed_precision: bool = True


@dataclass
class LossConfig:
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    focal_weight: float = 0.0
    focal_gamma: float = 2.0
    ignore_index: int = 255


@dataclass
class AugmentationConfig:
    horizontal_flip: bool = True
    vertical_flip: bool = True
    random_rotate: bool = True
    random_crop: bool = False
    color_jitter: bool = False
    normalize: bool = True


@dataclass
class FloodSenseConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    output_dir: str = "./outputs"
    seed: int = 42
    device: str = "cuda"


def load_env_local(project_root: Path) -> Dict[str, str]:
    env_path = project_root / ".env.local"
    env_vars = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
    return env_vars


def load_config(config_path: str) -> FloodSenseConfig:
    config_path = Path(config_path)
    project_root = config_path.parent.parent

    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    env_vars = load_env_local(project_root)

    dataset_cfg = DatasetConfig(**yaml_config.get('dataset', {}))
    preprocessing_cfg = PreprocessingConfig(**yaml_config.get('preprocessing', {}))
    model_cfg = ModelConfig(**yaml_config.get('model', {}))

    training_dict = yaml_config.get('training', {})
    opt_dict = training_dict.pop('optimizer', {})
    sched_dict = training_dict.pop('scheduler', {})
    training_cfg = TrainingConfig(
        **training_dict,
        optimizer=OptimizerConfig(**opt_dict),
        scheduler=SchedulerConfig(**sched_dict)
    )

    loss_cfg = LossConfig(**yaml_config.get('loss', {}))
    aug_cfg = AugmentationConfig(**yaml_config.get('augmentation', {}))

    # Apply .env.local overrides
    dataset_name_lower = dataset_cfg.name.lower()
    if dataset_name_lower in ['s1gfloods', 's1g'] and 'S1GFLOODS_ROOT' in env_vars:
        dataset_cfg.root = env_vars['S1GFLOODS_ROOT']
    elif dataset_name_lower == 'sen1floods11' and 'SEN1FLOODS11_ROOT' in env_vars:
        dataset_cfg.root = env_vars['SEN1FLOODS11_ROOT']
    elif dataset_name_lower == 'ombrias1' and 'OMBRIAS1_ROOT' in env_vars:
        dataset_cfg.root = env_vars['OMBRIAS1_ROOT']
    elif 'DATASET_ROOT' in env_vars:
        dataset_cfg.root = env_vars['DATASET_ROOT']

    if 'BATCH_SIZE' in env_vars:
        training_cfg.batch_size = int(env_vars['BATCH_SIZE'])
    if 'NUM_WORKERS' in env_vars:
        training_cfg.num_workers = int(env_vars['NUM_WORKERS'])
    if 'EPOCHS' in env_vars:
        training_cfg.epochs = int(env_vars['EPOCHS'])

    output_dir = env_vars.get('OUTPUT_DIR', yaml_config.get('output_dir', './outputs'))

    return FloodSenseConfig(
        dataset=dataset_cfg,
        preprocessing=preprocessing_cfg,
        model=model_cfg,
        training=training_cfg,
        loss=loss_cfg,
        augmentation=aug_cfg,
        output_dir=output_dir,
        seed=yaml_config.get('seed', 42),
        device=yaml_config.get('device', 'cuda')
    )
