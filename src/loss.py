"""
Loss Functions for Handling Severe Class Imbalance
Multi-Class Coconut Tree Disease Detection & Classification

Imbalance in Thite et al. (2023):
- Gray Leaf Spot: 2,135 (36.8%)
- Leaf Rot: 1,673 (28.9%)
- Stem Bleeding: 1,006 (17.3%)
- Bud Root Dropping: 514 (8.9%)
- Bud Rot: 470 (8.1%)

Ratio of Majority to Minority is ~ 4.5 : 1
"""

from typing import Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import config
except ImportError:
    from src import config


def compute_class_weights(
    class_counts: Union[Dict[str, int], List[int], np.ndarray],
    method: str = "inverse_freq",
    beta: float = 0.999,
) -> torch.Tensor:
    """
    Computes class weights to penalize errors on minority classes more heavily.
    
    Methods:
    - 'inverse_freq': w_c = Total_Samples / (Num_Classes * Count_c)
      Standard, intuitive, balanced penalty.
    - 'effective_samples': Cui et al. (CVPR 2019)
      w_c = (1 - beta) / (1 - beta^Count_c), best for large imbalanced vision datasets.
    - 'sqrt_inverse': w_c = 1 / sqrt(Count_c), gentler weighting when dataset is noisy.
    """
    if isinstance(class_counts, dict):
        counts = np.array([class_counts[cls] for cls in config.CLASSES], dtype=np.float32)
    else:
        counts = np.array(class_counts, dtype=np.float32)

    num_classes = len(counts)
    total_samples = np.sum(counts)

    if method == "inverse_freq":
        weights = total_samples / (num_classes * counts)
    elif method == "effective_samples":
        # Effective number of samples: E_n = (1 - beta^n) / (1 - beta)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        # Normalize so mean weight is 1.0
        weights = weights / np.mean(weights)
    elif method == "sqrt_inverse":
        weights = 1.0 / np.sqrt(counts)
        weights = weights / np.mean(weights)
    else:
        raise ValueError(f"Unknown weight method: {method}")

    # Normalize weights to have mean of 1.0 so learning rate scale is preserved
    weights = weights / np.sum(weights) * num_classes
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss (Lin et al., ICCV 2017)
    
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Focuses training on hard minority examples (Bud Rot, Bud Root Dropping)
    by down-weighting the loss from well-classified easy majority examples
    (Gray Leaf Spot, Leaf Rot).
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha  # Class weights tensor (shape: [num_classes])
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: raw logits [batch_size, num_classes]
        targets: ground truth class indices [batch_size]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of true class

        focal_term = (1.0 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_function(
    loss_type: str = config.LOSS_TYPE,
    class_weights: Optional[torch.Tensor] = None,
    device: str = config.DEVICE,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """
    Factory function returning the configured loss criterion with optional label smoothing.
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)

    if loss_type == "standard_ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    elif loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=config.FOCAL_GAMMA)
    else:
        raise ValueError(
            f"Unsupported loss_type '{loss_type}'. Choose from: 'standard_ce', 'weighted_ce', 'focal'"
        )


if __name__ == "__main__":
    print("Testing Loss & Weight Calculations with Benchmark Counts:")
    weights = compute_class_weights(config.BENCHMARK_CLASS_COUNTS, method="inverse_freq")
    for cls_name, w in zip(config.CLASSES, weights):
        print(f"  - {cls_name:20s}: weight = {w.item():.3f} (Count: {config.BENCHMARK_CLASS_COUNTS[cls_name]})")

    # Quick forward pass test
    dummy_logits = torch.randn(4, config.NUM_CLASSES)
    dummy_targets = torch.tensor([0, 1, 2, 4])
    criterion = get_loss_function("focal", class_weights=weights, device="cpu")
    loss_val = criterion(dummy_logits, dummy_targets)
    print(f"Test Focal Loss value: {loss_val.item():.4f}")
