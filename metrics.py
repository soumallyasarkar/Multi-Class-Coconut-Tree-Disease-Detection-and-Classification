"""
Evaluation Metrics Tailored for Imbalanced Classification
Multi-Class Coconut Tree Disease Detection & Classification

Calculates:
- Overall Accuracy
- Macro-Averaged Precision, Recall, and F1-Score (Equal weight to all classes)
- Per-Class Precision, Recall, F1-Score
- Multi-Class Confusion Matrix
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5
) -> np.ndarray:
    """
    Computes confusion matrix where:
    rows = Ground Truth class
    columns = Predicted class
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def compute_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    num_classes: int = 5,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Computes comprehensive classification metrics.
    Macro-averaged metrics treat every disease class with equal importance,
    which is critical for identifying failures on minority classes (Bud Rot).
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    total_samples = len(y_true)

    # Overall Accuracy
    accuracy = np.trace(cm) / max(total_samples, 1)

    # Per-class calculations
    per_class = {}
    precisions, recalls, f1s = [], [], []
    supports = []

    for i in range(num_classes):
        cls_name = class_names[i] if class_names and i < len(class_names) else f"Class_{i}"
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

        per_class[cls_name] = {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "support": int(support),
        }

    # Macro averages (unweighted average across classes)
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    # Weighted averages (weighted by class support)
    total_support = max(sum(supports), 1)
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total_support)

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": cm,
        "per_class": per_class,
    }
