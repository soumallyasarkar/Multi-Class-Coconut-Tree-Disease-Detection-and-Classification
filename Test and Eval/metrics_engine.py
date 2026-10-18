"""
Statistical and Calibration Metrics Engine
Multi-Class Coconut Tree Disease Detection & Classification

Implements:
1. Macro-F1 with 95% Bootstrap Confidence Intervals (1,000 iterations)
2. Expected Calibration Error (ECE) with Adaptive/Equal-width Bins
3. Brier Multi-class Calibration Score
4. Temperature Scaling for Probability Calibration (Guo et al.)
5. Selective Classification / Rejection Curves (Accuracy vs Coverage)
6. AUROC for Error and Failure Detection
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score


def compute_macro_f1_with_bootstrap_ci(
    targets: np.ndarray,
    preds: np.ndarray,
    num_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> dict:
    """
    Computes Macro F1-Score along with empirical bootstrap confidence intervals.
    
    Returns:
        dict: {"macro_f1": float, "ci_lower": float, "ci_upper": float}
    """
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    n = len(targets)
    
    base_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    
    if n == 0:
        return {"macro_f1": base_f1, "ci_lower": base_f1, "ci_upper": base_f1}
        
    rng = np.random.default_rng(seed)
    bootstrapped_scores = []
    
    for _ in range(num_bootstraps):
        indices = rng.integers(0, n, size=n)
        sample_targets = targets[indices]
        sample_preds = preds[indices]
        score = f1_score(sample_targets, sample_preds, average="macro", zero_division=0)
        bootstrapped_scores.append(score)
        
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(bootstrapped_scores, alpha * 100))
    upper = float(np.percentile(bootstrapped_scores, (1.0 - alpha) * 100))
    
    return {
        "macro_f1": round(base_f1, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
    }


def compute_ece(probs: np.ndarray, targets: np.ndarray, num_bins: int = 15) -> float:
    """
    Computes the Expected Calibration Error (ECE) across confidence bins.
    
    Args:
        probs: (N, C) softmax predicted probabilities
        targets: (N,) ground-truth class indices
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(targets)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(round(ece, 4))


def compute_brier_score(probs: np.ndarray, targets: np.ndarray, num_classes: int = 5) -> float:
    """
    Computes the multi-class Brier score: mean squared error between probabilities and one-hot targets.
    Lower is better (0.0 = perfect probabilistic forecast).
    """
    n = len(targets)
    if n == 0:
        return 0.0
    one_hot = np.zeros((n, num_classes), dtype=np.float32)
    one_hot[np.arange(n), targets] = 1.0
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(round(brier, 4))


class TemperatureScaler(nn.Module):
    """
    A post-processing module that learns a single scalar temperature T > 0
    to calibrate model logits using CrossEntropyLoss on validation data (Guo et al., 2017).
    """

    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by temperature."""
        temperature = self.temperature.clamp(min=1e-3)
        return logits / temperature

    def fit(self, val_logits: torch.Tensor, val_targets: torch.Tensor, max_iter: int = 100, lr: float = 0.01) -> float:
        """
        Finds optimal temperature T minimizing NLL on validation set.
        """
        self.to(val_logits.device)
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_targets)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        optimal_t = float(self.temperature.detach().cpu().item())
        return round(optimal_t, 4)


def compute_risk_coverage(probs: np.ndarray, targets: np.ndarray, coverages: list = None) -> dict:
    """
    Evaluates Selective Classification (Accuracy vs. Coverage).
    Images with lowest confidence are rejected first.
    
    Args:
        coverages: List of coverage proportions, e.g. [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    """
    if coverages is None:
        coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == targets)

    sorted_indices = np.argsort(-confidences)  # Descending order of confidence
    n = len(targets)
    results = {}

    for cov in coverages:
        k = max(1, int(round(cov * n)))
        selected_indices = sorted_indices[:k]
        acc_at_cov = float(np.mean(correct[selected_indices]))
        results[f"{int(cov * 100)}%"] = round(acc_at_cov, 4)

    return results


def compute_error_detection_auroc(confidences: np.ndarray, targets: np.ndarray, preds: np.ndarray) -> float:
    """
    Computes AUROC for spotting wrong predictions using prediction uncertainty (1 - confidence).
    0.5 = random guess, 1.0 = perfect error detector.
    """
    is_error = (preds != targets).astype(int)
    if len(np.unique(is_error)) < 2:
        return 0.5  # Cannot compute AUROC if no errors or only errors exist
        
    uncertainty_score = -confidences  # Higher uncertainty = higher risk of error
    return float(round(roc_auc_score(is_error, uncertainty_score), 4))
