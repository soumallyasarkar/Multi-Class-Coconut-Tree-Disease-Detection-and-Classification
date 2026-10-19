"""
High-Resolution Grad-CAM Visual Exhibits Generator (Matching Reference Report)
Multi-Class Coconut Tree Disease Detection & Classification

Generates:
1. Figure 8: Clean Grad-CAM comparison grid for all 5 disease categories across models
   [Original | ResNet50 | EfficientNet-B0 | MobileNetV2 | DenseNet-121]
2. Figure 9: Progressive Image Damage Breakdown for a diseased sample
   [Clean | Blur 2 | Blur 3 | JPEG 3 | Dark 3] across models

Saves outputs to: outputs/plots/report/
"""

import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "Test and Eval"))

from src import config, models
from corruptions import corrupt_image


class GradCAM:
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
        self.model.eval()
        self.gradients = None
        self.activations = None

        logits = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=False)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(input_tensor.size(2), input_tensor.size(3)), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

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


def get_target_layer(model: torch.nn.Module, model_name: str):
    model_name = model_name.lower()
    if "resnet" in model_name:
        return model.layer4[-1]
    elif "mobilenet" in model_name:
        return model.features[-1]
    elif "efficientnet" in model_name:
        return model.features[-1]
    elif "densenet" in model_name:
        return model.features.denseblock4
    else:
        return model.features[-1]


def overlay_cam_on_image(img_pil: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlays Grad-CAM jet heatmap on the original RGB image."""
    img_resized = np.array(img_pil.resize(config.IMAGE_SIZE, Image.BILINEAR)) / 255.0
    cmap = plt.cm.jet
    heatmap_colored = cmap(cam)[:, :, :3]
    overlay = (1.0 - alpha) * img_resized + alpha * heatmap_colored
    return np.clip(overlay, 0.0, 1.0)


def load_all_models(device):
    loaded = {}
    model_names = ["resnet50", "efficientnet_b0", "mobilenet_v2", "densenet121"]
    for m in model_names:
        ckpt_path = config.CHECKPOINT_DIR / f"best_{m}.pth"
        if ckpt_path.exists():
            net = models.build_model(m, num_classes=len(config.CLASSES), pretrained=False)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            net.load_state_dict(sd)
            net.to(device)
            net.eval()
            loaded[m] = net
            print(f"Loaded {m} for Grad-CAM generation.")
    return loaded


def generate_fig8_and_fig9(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaded_models = load_all_models(device)
    if not loaded_models:
        print("No models loaded. Aborting.")
        return

    model_keys = ["resnet50", "efficientnet_b0", "mobilenet_v2", "densenet121"]
    model_keys = [m for m in model_keys if m in loaded_models]

    grad_cams = {m: GradCAM(loaded_models[m], get_target_layer(loaded_models[m], m)) for m in model_keys}

    norm_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # -------------------------------------------------------------
    # Figure 8: Clean Grad-CAM Grid for Each Disease Category
    # -------------------------------------------------------------
    print("\nGenerating Figure 8: Clean Grad-CAM Grid...")
    # Find one representative image per class
    sample_images = {}
    for cls_name in config.CLASSES:
        cls_dir = config.RAW_DATA_DIR / cls_name
        if cls_dir.exists():
            imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.png"))
            if imgs:
                # pick a middle-indexed representative photo
                sample_images[cls_name] = imgs[len(imgs) // 3]

    num_rows = len(sample_images)
    num_cols = 1 + len(model_keys)  # original + models

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.2 * num_cols, 3.2 * num_rows))
    if num_rows == 1:
        axes = np.expand_dims(axes, 0)

    # Column titles
    col_titles = ["Original"] + [m for m in model_keys]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold", pad=8)

    for row_idx, (cls_name, img_path) in enumerate(sample_images.items()):
        cls_target = config.CLASSES.index(cls_name)
        img_pil = Image.open(img_path).convert("RGB")
        tensor_in = norm_transform(img_pil).unsqueeze(0).to(device)

        # 1. Original image in column 0
        resized_orig = img_pil.resize(config.IMAGE_SIZE, Image.BILINEAR)
        axes[row_idx, 0].imshow(resized_orig)
        axes[row_idx, 0].set_ylabel(cls_name, fontsize=11, fontweight="bold", rotation=90, labelpad=8)
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        # 2. Each model's overlay
        for m_idx, m_name in enumerate(model_keys):
            g_cam = grad_cams[m_name]
            cam = g_cam.generate_heatmap(tensor_in, target_class=cls_target)
            overlay = overlay_cam_on_image(img_pil, cam, alpha=0.55)

            ax = axes[row_idx, m_idx + 1]
            ax.imshow(overlay)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Figure 8: Clean Grad-CAM Visualizations across Candidate Models", fontsize=15, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig8_path = output_dir / "fig8_clean_cam_grid.png"
    fig.savefig(fig8_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 8 to: {fig8_path}")

    # -------------------------------------------------------------
    # Figure 9: Progressive Corruption Breakdown (Single Disease Image)
    # -------------------------------------------------------------
    print("\nGenerating Figure 9: Progressive Corruption Breakdown...")
    # Select a Stem Bleeding trunk image or Bud Rot crown image
    chosen_path = None
    for cand_class in ["Stem Bleeding", "Bud Rot", "Gray Leaf Spot"]:
        c_dir = config.RAW_DATA_DIR / cand_class
        if c_dir.exists():
            c_imgs = list(c_dir.glob("*.jpg"))
            if c_imgs:
                chosen_path = c_imgs[10] if len(c_imgs) > 10 else c_imgs[0]
                chosen_class = cand_class
                break

    if chosen_path is not None:
        target_idx = config.CLASSES.index(chosen_class)
        raw_pil = Image.open(chosen_path).convert("RGB")

        corruption_steps = [
            ("Clean", raw_pil),
            ("Blur 2", corrupt_image(raw_pil, "blur", 2)),
            ("Blur 3", corrupt_image(raw_pil, "blur", 3)),
            ("JPEG 3", corrupt_image(raw_pil, "jpeg", 3)),
            ("Dark 3", corrupt_image(raw_pil, "dark", 3)),
        ]

        fig, axes = plt.subplots(len(model_keys), len(corruption_steps), figsize=(3.2 * len(corruption_steps), 3.2 * len(model_keys)))
        if len(model_keys) == 1:
            axes = np.expand_dims(axes, 0)

        for col_idx, (c_label, _) in enumerate(corruption_steps):
            axes[0, col_idx].set_title(c_label, fontsize=12, fontweight="bold", pad=8)

        for row_idx, m_name in enumerate(model_keys):
            g_cam = grad_cams[m_name]
            axes[row_idx, 0].set_ylabel(m_name, fontsize=11, fontweight="bold", labelpad=8)

            for col_idx, (c_label, c_img) in enumerate(corruption_steps):
                c_tensor = norm_transform(c_img).unsqueeze(0).to(device)
                cam = g_cam.generate_heatmap(c_tensor, target_class=target_idx)
                overlay = overlay_cam_on_image(c_img, cam, alpha=0.55)

                ax = axes[row_idx, col_idx]
                ax.imshow(overlay)
                ax.set_xticks([])
                ax.set_yticks([])

        fig.suptitle(f"Figure 9: Grad-CAM Stability under Progressive Damage ({chosen_class})", fontsize=15, fontweight="bold", y=1.005)
        fig.tight_layout()
        fig9_path = output_dir / "fig9_cam_corruption_breakdown.png"
        fig.savefig(fig9_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Figure 9 to: {fig9_path}")

    # Cleanup hooks
    for g in grad_cams.values():
        g.remove_hooks()


if __name__ == "__main__":
    out_dir = config.OUTPUT_DIR / "plots" / "report"
    generate_fig8_and_fig9(out_dir)
