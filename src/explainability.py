"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) for Disease Symptom Localization
Multi-Class Coconut Tree Disease Detection & Classification

Addresses the research question:
"Classification tells us what disease is present; can we localize the affected region?"
Grad-CAM highlights the exact lesion/blight regions on leaves, buds, and bark
that influenced the CNN's decision.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    from . import config, dataset, models
except ImportError:
    from src import config, dataset, models


class GradCAM:
    """
    Computes Grad-CAM heatmaps for any candidate CNN architecture.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        """
        input_tensor: [1, 3, 224, 224]
        Returns:
            heatmap: 2D numpy array [224, 224] normalized between 0 and 1
            predicted_class: class index
            confidence: softmax probability
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = torch.argmax(probs, dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Global average pooling on gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H, W]

        # Apply ReLU to focus only on features that positively contribute
        cam = F.relu(cam)

        # Upsample to match input image resolution
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, confidence


def get_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """Finds the last convolutional layer for each backbone architecture."""
    model_name = model_name.lower()
    if "resnet" in model_name:
        return model.layer4[-1]
    elif "mobilenet" in model_name:
        return model.features[-1]
    elif "efficientnet" in model_name:
        return model.features[-1]
    elif "densenet" in model_name:
        return model.features.norm5
    elif "vgg" in model_name:
        return model.features[-1]
    else:
        raise ValueError(f"Target layer not defined for {model_name}")


def explain_image(
    image_path: Union[str, Path],
    checkpoint_path: Path,
    save_path: Optional[Path] = None,
) -> Tuple[str, float]:
    """
    Runs Grad-CAM visual explanation on a single image and generates a 3-panel figure:
    [Original Image | Heatmap | Overlay with Diagnosis]
    """
    image_path = Path(image_path)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model_name = checkpoint.get("model_name", "resnet50")

    model = models.build_model(model_name, num_classes=config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.DEVICE)
    model.eval()

    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)

    # Load and preprocess image
    orig_img = Image.open(image_path).convert("RGB")
    val_transforms = dataset.get_transforms("val")
    input_tensor = val_transforms(orig_img).unsqueeze(0).to(config.DEVICE)

    # Generate CAM
    cam, pred_idx, confidence = grad_cam.generate(input_tensor)
    pred_label = config.CLASSES[pred_idx]

    # Resize original image to 224x224 for overlay
    resized_orig = np.array(orig_img.resize(config.IMAGE_SIZE, Image.BILINEAR)) / 255.0

    # Color map
    cmap = plt.cm.jet
    heatmap_colored = cmap(cam)[:, :, :3]  # drop alpha

    # Alpha blending: 60% original image + 40% heatmap
    overlay = 0.6 * resized_orig + 0.4 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(resized_orig)
    axes[0].set_title(f"Input: {image_path.name}")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM Disease Activation")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Diagnosis: {pred_label}\nConfidence: {confidence * 100:.2f}%")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Explanation figure saved to: {save_path}")
    plt.close()

    return pred_label, confidence
