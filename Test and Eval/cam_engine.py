"""
Grad-CAM Explainability and Heatmap Stability Engine
Multi-Class Coconut Tree Disease Detection & Classification

Implements:
1. Native PyTorch Hook-based Grad-CAM for all 4 architectures
2. Pearson Correlation, SSIM, and Top-20% Overlap (IoU) between heatmaps
3. Cross-Model Agreement Analysis
4. Flip-Consistency Score for Unsupervised Failure/Error Detection
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F


def get_target_layer(model: torch.nn.Module, model_name: str):
    """Retrieves the target convolutional feature layer for Grad-CAM."""
    model_name = model_name.lower()
    if "resnet" in model_name:
        return model.layer4[-1]
    elif "mobilenet" in model_name:
        return model.features[-1]
    elif "efficientnet" in model_name:
        return model.features[-1]
    elif "densenet" in model_name:
        return model.features.denseblock4
    elif "vgg" in model_name:
        return model.features[-1]
    else:
        # Fallback to the last submodule in features if present
        if hasattr(model, "features"):
            return model.features[-1]
        raise ValueError(f"Target layer not defined for model architecture: {model_name}")


class GradCAM:
    """
    Native PyTorch Grad-CAM implementation.
    Registers forward and backward hooks on the last convolutional layer.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generates a 2D normalized Grad-CAM heatmap [0.0, 1.0] for the input tensor.
        
        Args:
            input_tensor: (1, C, H, W) torch Tensor
            target_class: Target class index. If None, uses argmax predicted class.
        """
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Forward pass
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured by hooks.")

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Upsample to input tensor resolution
        cam = F.interpolate(cam, size=(input_tensor.size(2), input_tensor.size(3)), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ==========================================
# Heatmap Quantitative Similarity Metrics
# ==========================================

def compute_pearson_correlation(cam1: np.ndarray, cam2: np.ndarray) -> float:
    """Computes 2D Pearson correlation coefficient between two heatmaps."""
    flat1 = cam1.flatten().astype(np.float64)
    flat2 = cam2.flatten().astype(np.float64)
    std1, std2 = np.std(flat1), np.std(flat2)
    if std1 < 1e-7 or std2 < 1e-7:
        return 0.0
    r = np.corrcoef(flat1, flat2)[0, 1]
    return float(round(r, 4)) if not np.isnan(r) else 0.0


def compute_ssim(cam1: np.ndarray, cam2: np.ndarray) -> float:
    """
    Computes Structural Similarity Index (SSIM) between two heatmaps.
    Simplified implementation for single-channel float heatmaps [0, 1].
    """
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    mu1 = cv2.GaussianBlur(cam1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(cam2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(cam1 * cam1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(cam2 * cam2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(cam1 * cam2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(round(np.mean(ssim_map), 4))


def compute_top_k_overlap(cam1: np.ndarray, cam2: np.ndarray, top_fraction: float = 0.20) -> float:
    """
    Computes Intersection-over-Union (IoU) of the top K% most salient pixels.
    """
    flat1 = cam1.flatten()
    flat2 = cam2.flatten()
    k = max(1, int(len(flat1) * top_fraction))

    thresh1 = np.partition(flat1, -k)[-k]
    thresh2 = np.partition(flat2, -k)[-k]

    mask1 = (cam1 >= thresh1)
    mask2 = (cam2 >= thresh2)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 1.0
    return float(round(intersection / union, 4))


def compute_flip_consistency(grad_cam: GradCAM, input_tensor: torch.Tensor, target_class: int) -> float:
    """
    Tests explanation stability against geometric mirroring (horizontal flip).
    Generates CAM on original, flips input tensor horizontally, generates CAM on flipped,
    then flips the second CAM back horizontally and computes Pearson correlation.
    
    Lower correlation indicates high instability / potential prediction failure.
    """
    # 1. CAM on original image
    cam_orig = grad_cam.generate_heatmap(input_tensor, target_class=target_class)

    # 2. Horizontal flip tensor
    flipped_tensor = torch.flip(input_tensor, dims=[3])

    # 3. CAM on flipped image
    cam_flipped = grad_cam.generate_heatmap(flipped_tensor, target_class=target_class)

    # 4. Flip back CAM
    cam_flipped_back = np.fliplr(cam_flipped)

    # 5. Measure similarity
    return compute_pearson_correlation(cam_orig, cam_flipped_back)
