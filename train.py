"""
FloodSense Training Script

Usage:
    python train.py --config configs/s1gfloods.yaml
    python train.py --config configs/sen1floods11.yaml
    python train.py --config configs/ombrias1.yaml
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.model import FloodSense, FloodSenseModelConfig
from src.losses import build_loss
from src.metrics import FloodMetrics
from data import build_dataloader, infer_input_channels


def setup_logging(output_dir: str) -> logging.Logger:
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def create_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    opt_config = config.training.optimizer
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'encoder' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    param_groups = [
        {'params': backbone_params, 'lr': opt_config.lr * 0.1},
        {'params': other_params, 'lr': opt_config.lr},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=opt_config.weight_decay)


def create_scheduler(optimizer, config, steps_per_epoch: int):
    sched_config = config.training.scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=sched_config.warmup_epochs * steps_per_epoch)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(config.training.epochs - sched_config.warmup_epochs) * steps_per_epoch, eta_min=sched_config.min_lr)
    return SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[sched_config.warmup_epochs * steps_per_epoch])


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, config, logger, writer, device):
    model.train()
    total_loss = 0.0
    loss_components = {}
    num_batches = len(train_loader)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True, dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
        img_pre = batch['pre'].to(device)
        img_post = batch['post'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        with autocast('cuda', enabled=config.training.mixed_precision):
            outputs = model(img_pre, img_post)
            loss, loss_dict = criterion(outputs, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf loss at batch {batch_idx}. Skipping.")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0.0) + v

        pbar.set_postfix({'loss': f'{total_loss / (batch_idx + 1):.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})

    for k in loss_components:
        loss_components[k] /= num_batches
    return {'loss': total_loss / num_batches, **loss_components}


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    metrics = FloodMetrics()
    total_loss = 0.0
    pbar = tqdm(val_loader, desc="Validating", leave=False, dynamic_ncols=True)

    for batch in pbar:
        img_pre = batch['pre'].to(device)
        img_post = batch['post'].to(device)
        labels = batch['label'].to(device)

        outputs = model(img_pre, img_post)
        loss, _ = criterion(outputs, labels)
        total_loss += loss.item()

        probs = torch.softmax(outputs['logits'], dim=1)
        preds = probs.argmax(dim=1)
        metrics.update(preds.cpu(), labels.cpu(), probs.cpu())
        pbar.set_postfix({'loss': f'{total_loss / (pbar.n + 1):.4f}'})

    results = metrics.compute()
    return {'loss': total_loss / len(val_loader), **results}


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, output_dir, is_best=False, config=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    torch.save(checkpoint, os.path.join(output_dir, 'latest.pth'))
    if is_best:
        torch.save(checkpoint, os.path.join(output_dir, 'best.pth'))
    if (epoch + 1) % 10 == 0:
        torch.save(checkpoint, os.path.join(output_dir, f'epoch_{epoch+1}.pth'))


def main():
    parser = argparse.ArgumentParser(description='Train FloodSense')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(config.output_dir, f'{config.dataset.name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {device}")

    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))

    preprocessing_config = config.preprocessing.__dict__
    effective_in_channels = infer_input_channels(
        config.dataset.name,
        config.dataset.in_channels,
        preprocessing_config
    )

    train_loader = build_dataloader(
        config.dataset.name,
        config.dataset.root,
        'train',
        config.training.batch_size,
        num_workers=config.training.num_workers,
        image_size=config.dataset.image_size,
        in_channels=config.dataset.in_channels,
        preprocessing_config=preprocessing_config
    )
    val_loader = build_dataloader(
        config.dataset.name,
        config.dataset.root,
        'val',
        config.training.batch_size,
        num_workers=config.training.num_workers,
        image_size=config.dataset.image_size,
        in_channels=config.dataset.in_channels,
        preprocessing_config=preprocessing_config
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    model_config = FloodSenseModelConfig(
        in_channels=effective_in_channels, num_classes=config.dataset.num_classes,
        encoder=config.model.encoder, pretrained=config.model.pretrained,
        hidden_dim=config.model.hidden_dim, lstm_hidden=config.model.lstm_hidden,
        lstm_layers=config.model.lstm_layers, use_attention=config.model.use_attention,
        dropout=config.model.dropout, img_size=config.dataset.image_size
    )

    model = FloodSense(model_config).to(device)
    logger.info(f"Model parameters: {model.get_num_parameters():,}")

    criterion = build_loss({
        'bce_weight': config.loss.bce_weight, 'dice_weight': config.loss.dice_weight,
        'focal_weight': config.loss.focal_weight, 'focal_gamma': config.loss.focal_gamma,
        'ignore_index': config.loss.ignore_index
    })

    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    scaler = GradScaler('cuda', enabled=config.training.mixed_precision)

    start_epoch, best_metric = 0, 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['metrics'].get('F1', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}")

    logger.info("Starting training...")
    for epoch in range(start_epoch, config.training.epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, config, logger, writer, device)
        val_metrics = validate(model, val_loader, criterion, device)

        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f} | F1: {val_metrics['F1']:.4f} | IoU: {val_metrics['IoU']:.4f}")

        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Metrics/F1', val_metrics['F1'], epoch)
        writer.add_scalar('Metrics/IoU', val_metrics['IoU'], epoch)

        is_best = val_metrics['F1'] > best_metric
        if is_best:
            best_metric = val_metrics['F1']
            logger.info(f"New best F1: {best_metric:.4f}")
        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, output_dir, is_best, config)

    logger.info(f"\nTraining complete! Best F1: {best_metric:.4f}")
    writer.close()


if __name__ == '__main__':
    main()
