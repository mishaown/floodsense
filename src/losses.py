"""
Loss Functions for FloodSense

BCE + Dice + Focal loss combination for binary segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: [B, 2, H, W] logits or [B, H, W] probabilities
            targets: [B, H, W] ground truth labels

        Returns:
            Dice loss scalar
        """
        if predictions.dim() == 4:
            probs = F.softmax(predictions, dim=1)[:, 1]  # Flood class
        else:
            probs = predictions

        # Create valid mask
        valid_mask = targets != self.ignore_index
        probs = probs[valid_mask]
        targets_valid = targets[valid_mask].float()

        if probs.numel() == 0:
            return torch.tensor(0.0, device=predictions.device)

        intersection = (probs * targets_valid).sum()
        union = probs.sum() + targets_valid.sum()

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = 255,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: [B, 2, H, W] logits
            targets: [B, H, W] ground truth labels

        Returns:
            Focal loss scalar
        """
        if predictions.dim() == 4:
            B, C, H, W = predictions.shape
            predictions = predictions.permute(0, 2, 3, 1).reshape(-1, C)
            targets = targets.reshape(-1)

        # Filter valid pixels
        valid_mask = targets != self.ignore_index
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        if predictions.numel() == 0:
            return torch.tensor(0.0, device=predictions.device)

        # Compute focal loss
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        probs = F.softmax(predictions, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice Loss.

    Often used for medical/remote sensing segmentation.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        ignore_index: int = 255
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

        self.bce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: [B, 2, H, W] logits
            targets: [B, H, W] ground truth

        Returns:
            total_loss, loss_dict
        """
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)

        total = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        loss_dict = {
            'bce': bce_loss.item(),
            'dice': dice_loss.item(),
            'total': total.item()
        }

        return total, loss_dict


class FloodLoss(nn.Module):
    """
    Complete loss function for FloodSense.

    Combines:
    - BCE/Cross-Entropy loss
    - Dice loss
    - Optional Focal loss
    - Optional magnitude loss
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.0,
        focal_gamma: float = 2.0,
        mag_weight: float = 0.5,
        ignore_index: int = 255
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.mag_weight = mag_weight
        self.ignore_index = ignore_index

        # Loss components
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

        if focal_weight > 0:
            self.focal = FocalLoss(gamma=focal_gamma, ignore_index=ignore_index)
        else:
            self.focal = None

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        target_magnitude: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            outputs: Model outputs with 'logits' and optionally 'magnitude'
            targets: [B, H, W] ground truth labels
            target_magnitude: [B, H, W] optional magnitude targets

        Returns:
            total_loss, loss_dict
        """
        logits = outputs['logits']
        loss_dict = {}

        # BCE/Cross-Entropy loss
        ce_loss = self.ce(logits, targets)
        loss_dict['ce'] = ce_loss.item()

        # Dice loss
        dice_loss = self.dice(logits, targets)
        loss_dict['dice'] = dice_loss.item()

        # Total segmentation loss
        total = self.bce_weight * ce_loss + self.dice_weight * dice_loss

        # Focal loss
        if self.focal is not None and self.focal_weight > 0:
            focal_loss = self.focal(logits, targets)
            loss_dict['focal'] = focal_loss.item()
            total = total + self.focal_weight * focal_loss

        # Magnitude loss
        if 'magnitude' in outputs and self.mag_weight > 0:
            magnitude = outputs['magnitude'].squeeze(1)

            # Create magnitude target from labels
            if target_magnitude is None:
                target_magnitude = (targets > 0).float()
                target_magnitude[targets == self.ignore_index] = 0

            valid_mask = targets != self.ignore_index
            if valid_mask.any():
                mag_loss = F.mse_loss(
                    magnitude[valid_mask],
                    target_magnitude[valid_mask]
                )
                loss_dict['magnitude'] = mag_loss.item()
                total = total + self.mag_weight * mag_loss

        loss_dict['total'] = total.item()

        return total, loss_dict


def build_loss(config: Optional[Dict] = None) -> FloodLoss:
    """Factory function to build loss."""
    config = config or {}

    return FloodLoss(
        bce_weight=config.get('bce_weight', 1.0),
        dice_weight=config.get('dice_weight', 1.0),
        focal_weight=config.get('focal_weight', 0.0),
        focal_gamma=config.get('focal_gamma', 2.0),
        mag_weight=config.get('mag_weight', 0.5),
        ignore_index=config.get('ignore_index', 255)
    )
