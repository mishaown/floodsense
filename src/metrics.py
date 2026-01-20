"""
Flood Detection Metrics

Implements: Precision, Recall, F1, IoU, OA, Kappa, MCC, AUC-ROC, AP
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix as sk_confusion_matrix
)


class FloodMetrics:
    """
    Binary Flood Change Detection Metrics.

    Computes comprehensive evaluation metrics for flood detection,
    following standards from high-impact remote sensing journals.

    Args:
        threshold: Probability threshold for binary classification
        ignore_index: Label value to ignore (default: 255)
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        threshold: float = 0.5,
        ignore_index: int = 255,
        eps: float = 1e-7
    ):
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.eps = eps
        self.reset()

    def reset(self):
        """Reset all counters."""
        self.tp = 0  # True Positives (flood correctly detected)
        self.fp = 0  # False Positives (non-flood predicted as flood)
        self.tn = 0  # True Negatives (non-flood correctly identified)
        self.fn = 0  # False Negatives (flood missed)
        self.total = 0

        # For probabilistic metrics
        self.all_probs = []
        self.all_labels = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with batch predictions.

        Args:
            predictions: [B, H, W] binary predictions or [B, 2, H, W] logits
            targets: [B, H, W] ground truth labels (0=no-flood, 1=flood)
            probabilities: [B, H, W] or [B, 2, H, W] prediction probabilities
        """
        # Handle different input formats
        if predictions.dim() == 4:
            # [B, 2, H, W] -> [B, H, W]
            predictions = predictions.argmax(dim=1)

        preds = predictions.cpu().numpy().flatten()
        tgts = targets.cpu().numpy().flatten()

        # Handle probabilities
        if probabilities is not None:
            if probabilities.dim() == 4:
                probs = probabilities[:, 1, :, :].cpu().numpy().flatten()  # Flood class prob
            else:
                probs = probabilities.cpu().numpy().flatten()

        # Filter out ignored pixels
        valid_mask = tgts != self.ignore_index
        preds = preds[valid_mask]
        tgts = tgts[valid_mask]

        if probabilities is not None:
            probs = probs[valid_mask]
            self.all_probs.extend(probs.tolist())
            self.all_labels.extend(tgts.tolist())

        # Binarize
        preds = (preds > 0).astype(int)
        tgts = (tgts > 0).astype(int)

        # Update confusion matrix components
        self.tp += int(np.sum((preds == 1) & (tgts == 1)))
        self.fp += int(np.sum((preds == 1) & (tgts == 0)))
        self.tn += int(np.sum((preds == 0) & (tgts == 0)))
        self.fn += int(np.sum((preds == 0) & (tgts == 1)))
        self.total += len(preds)

    def compute_precision(self) -> float:
        """Compute precision = TP / (TP + FP)."""
        return self.tp / (self.tp + self.fp + self.eps)

    def compute_recall(self) -> float:
        """Compute recall (sensitivity) = TP / (TP + FN)."""
        return self.tp / (self.tp + self.fn + self.eps)

    def compute_f1(self) -> float:
        """Compute F1-Score = 2 * (Precision * Recall) / (Precision + Recall)."""
        precision = self.compute_precision()
        recall = self.compute_recall()
        return 2 * precision * recall / (precision + recall + self.eps)

    def compute_iou(self) -> float:
        """
        Compute IoU (Jaccard Index) for flood class.

        IoU = TP / (TP + FP + FN)
        """
        return self.tp / (self.tp + self.fp + self.fn + self.eps)

    def compute_mean_iou(self) -> float:
        """
        Compute mean IoU over both classes.

        mIoU = (IoU_no_flood + IoU_flood) / 2
        """
        iou_flood = self.compute_iou()
        iou_no_flood = self.tn / (self.tn + self.fp + self.fn + self.eps)
        return (iou_flood + iou_no_flood) / 2

    def compute_overall_accuracy(self) -> float:
        """Compute Overall Accuracy = (TP + TN) / Total."""
        return (self.tp + self.tn) / (self.total + self.eps)

    def compute_kappa(self) -> float:
        """
        Compute Cohen's Kappa coefficient.

        Kappa = (p_o - p_e) / (1 - p_e)

        where p_o is observed agreement, p_e is expected agreement by chance.
        """
        total = self.total + self.eps

        # Observed agreement
        p_o = (self.tp + self.tn) / total

        # Expected agreement
        p_flood_pred = (self.tp + self.fp) / total
        p_flood_true = (self.tp + self.fn) / total
        p_no_flood_pred = (self.tn + self.fn) / total
        p_no_flood_true = (self.tn + self.fp) / total

        p_e = p_flood_pred * p_flood_true + p_no_flood_pred * p_no_flood_true

        if p_e >= 1.0:
            return 0.0

        return (p_o - p_e) / (1 - p_e + self.eps)

    def compute_mcc(self) -> float:
        """
        Compute Matthews Correlation Coefficient.

        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

        Considered one of the best single metrics for binary classification.
        """
        numerator = float(self.tp * self.tn - self.fp * self.fn)
        denominator = np.sqrt(
            float(self.tp + self.fp) *
            float(self.tp + self.fn) *
            float(self.tn + self.fp) *
            float(self.tn + self.fn)
        )
        return numerator / (denominator + self.eps)

    def compute_specificity(self) -> float:
        """Compute specificity = TN / (TN + FP)."""
        return self.tn / (self.tn + self.fp + self.eps)

    def compute_balanced_accuracy(self) -> float:
        """Compute balanced accuracy = (Sensitivity + Specificity) / 2."""
        sensitivity = self.compute_recall()
        specificity = self.compute_specificity()
        return (sensitivity + specificity) / 2

    def compute_auc_roc(self) -> float:
        """
        Compute Area Under ROC Curve.

        Requires probability predictions.
        """
        if len(self.all_probs) == 0:
            return 0.0

        try:
            return roc_auc_score(self.all_labels, self.all_probs)
        except ValueError:
            return 0.0

    def compute_average_precision(self) -> float:
        """
        Compute Average Precision (area under precision-recall curve).

        Requires probability predictions.
        """
        if len(self.all_probs) == 0:
            return 0.0

        try:
            return average_precision_score(self.all_labels, self.all_probs)
        except ValueError:
            return 0.0

    def compute_dice(self) -> float:
        """
        Compute Dice coefficient (same as F1 for binary).

        Dice = 2 * TP / (2*TP + FP + FN)
        """
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn + self.eps)

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.

        Returns:
            2x2 array: [[TN, FP], [FN, TP]]
        """
        return np.array([
            [self.tn, self.fp],
            [self.fn, self.tp]
        ])

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary with all metric values.
        """
        return {
            # Primary metrics (for model selection)
            'F1': self.compute_f1(),
            'IoU': self.compute_iou(),
            'mIoU': self.compute_mean_iou(),

            # Standard classification metrics
            'Precision': self.compute_precision(),
            'Recall': self.compute_recall(),
            'Specificity': self.compute_specificity(),
            'OA': self.compute_overall_accuracy(),

            # Advanced metrics
            'Kappa': self.compute_kappa(),
            'MCC': self.compute_mcc(),
            'Dice': self.compute_dice(),
            'BalancedAcc': self.compute_balanced_accuracy(),

            # Probabilistic metrics (if available)
            'AUC-ROC': self.compute_auc_roc(),
            'AP': self.compute_average_precision(),
        }

    def compute_detailed(self) -> Dict[str, any]:
        """
        Compute detailed metrics with confusion matrix.

        Returns:
            Dictionary with all metrics and additional details.
        """
        metrics = self.compute()
        metrics['ConfusionMatrix'] = self.get_confusion_matrix()
        metrics['TotalPixels'] = self.total
        metrics['FloodPixels'] = self.tp + self.fn
        metrics['NonFloodPixels'] = self.tn + self.fp
        return metrics

    def __str__(self) -> str:
        """String representation of metrics."""
        metrics = self.compute()
        lines = [
            "=" * 50,
            "FloodSense Evaluation Metrics",
            "=" * 50,
            f"F1-Score:     {metrics['F1']:.4f}",
            f"IoU (Flood):  {metrics['IoU']:.4f}",
            f"mIoU:         {metrics['mIoU']:.4f}",
            f"Precision:    {metrics['Precision']:.4f}",
            f"Recall:       {metrics['Recall']:.4f}",
            f"OA:           {metrics['OA']:.4f}",
            f"Kappa:        {metrics['Kappa']:.4f}",
            f"MCC:          {metrics['MCC']:.4f}",
            "-" * 50,
        ]
        if metrics['AUC-ROC'] > 0:
            lines.append(f"AUC-ROC:      {metrics['AUC-ROC']:.4f}")
            lines.append(f"AP:           {metrics['AP']:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)


def compute_metrics_from_arrays(
    predictions: np.ndarray,
    targets: np.ndarray,
    ignore_index: int = 255
) -> Dict[str, float]:
    """
    Convenience function to compute metrics from numpy arrays.

    Args:
        predictions: Binary predictions
        targets: Ground truth labels
        ignore_index: Value to ignore

    Returns:
        Dictionary with metrics
    """
    metrics = FloodMetrics(ignore_index=ignore_index)
    metrics.update(
        torch.from_numpy(predictions),
        torch.from_numpy(targets)
    )
    return metrics.compute()
