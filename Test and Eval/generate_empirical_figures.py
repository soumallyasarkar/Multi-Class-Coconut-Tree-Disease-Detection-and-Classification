"""
Generate Authentic Empirical Figures from Trained Models & 13,836 Image Dataset
Multi-Class Coconut Tree Disease Detection & Classification

Generates:
1. Figure 5: Accuracy on answered images versus coverage under blur_3, jpeg_3, dark_3, bright_3
2. Figure 6: Reliability diagrams pooled over all conditions (Raw vs Temperature Scaled T_mixed)
3. Figure 3: Confusion matrices under blur_3 (top) and jpeg_3 (bottom)
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import config, models
from metrics_engine import TemperatureScaler, compute_ece

CLASSES = config.CLASSES
NUM_CLASSES = len(CLASSES)

MODEL_COLORS = {
    "resnet50": "#1f77b4",       # Blue
    "efficientnet_b0": "#ff7f0e",# Orange
    "mobilenet_v2": "#2ca02c",   # Green
    "densenet121": "#9467bd",    # Purple
}


def corrupt_pil(img: Image.Image, corruption: str, severity: int) -> Image.Image:
    """Fast PIL in-memory corruption."""
    if corruption == "blur":
        sigma = {1: 1.0, 2: 2.0, 3: 4.0}[severity]
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))
    elif corruption == "dark":
        factor = {1: 0.8, 2: 0.5, 3: 0.2}[severity]
        return ImageEnhance.Brightness(img).enhance(factor)
    elif corruption == "bright":
        factor = {1: 1.2, 2: 1.8, 3: 2.2}[severity]
        return ImageEnhance.Brightness(img).enhance(factor)
    elif corruption == "jpeg":
        quality = {1: 50, 2: 20, 3: 5}[severity]
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return img


def get_dataset_files(data_dir: Path) -> Tuple[List[Path], List[int]]:
    """Scans all 13,836 field images by class folder or prefix."""
    image_paths, targets = [], []

    # 1. Folder structure check
    for idx, c_name in enumerate(CLASSES):
        folder = data_dir / c_name
        if folder.exists() and folder.is_dir():
            files = [p for p in folder.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            image_paths.extend(files)
            targets.extend([idx] * len(files))

    # 2. Filename prefix check if folder check empty
    if len(image_paths) == 0:
        prefix_mapping = [
            ("BudRootDropping", 0),
            ("BudRot", 1),
            ("GrayLeafSpot", 2),
            ("LeafRot", 3),
            ("StemBleeding", 4),
        ]
        all_imgs = [p for p in data_dir.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        for img_path in all_imgs:
            name = img_path.name
            for prefix, label_idx in prefix_mapping:
                if name.startswith(prefix):
                    image_paths.append(img_path)
                    targets.append(label_idx)
                    break

    return image_paths, targets


class FastImageLoader(Dataset):
    def __init__(self, image_paths: List[Path], targets: List[int]):
        self.image_paths = image_paths
        self.targets = targets

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Open and pre-resize to 256x256 to speed up corruptions and transforms
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        return img, self.targets[idx]


def collate_and_corrupt(batch, condition: str, transform: transforms.Compose):
    """Corrupts and transforms a mini-batch of PIL images."""
    imgs, targets = zip(*batch)
    if condition == "clean":
        corrupted = imgs
    else:
        family, sev_str = condition.split("_")
        sev = int(sev_str)
        corrupted = [corrupt_pil(img, family, sev) for img in imgs]

    tensors = torch.stack([transform(img) for img in corrupted])
    target_tensor = torch.tensor(targets, dtype=torch.long)
    return tensors, target_tensor


def run_empirical_eval(full_eval: bool = False, sample_limit: int = None, batch_size: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"   COMPUTING EMPIRICAL FIGURES FROM TRAINED MODELS")
    print(f"   Device: {device} | Batch Size: {batch_size}")
    print(f"   Mode: {'Full (13 conditions)' if full_eval else 'Essential (5 conditions: clean, blur_3, jpeg_3, dark_3, bright_3)'}")
    print(f"=======================================================\n")

    data_dir = ROOT_DIR / "data" / "Coconut_Tree_Disease_Dataset.v3i.folder"
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    image_paths, targets = get_dataset_files(data_dir)
    if sample_limit is not None and sample_limit < len(image_paths):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(image_paths), size=sample_limit, replace=False)
        image_paths = [image_paths[i] for i in indices]
        targets = [targets[i] for i in indices]
        print(f"Sampled {len(image_paths)} images for fast empirical evaluation")

    num_images = len(image_paths)
    print(f"Found {num_images} images in {data_dir.name}")
    if num_images == 0:
        raise RuntimeError("No images found in dataset directory!")

    # Load 4 models
    model_names = ["mobilenet_v2", "resnet50", "efficientnet_b0", "densenet121"]
    loaded_models = {}
    for m_name in model_names:
        ckpt_path = config.CHECKPOINTS_DIR / f"best_{m_name}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"Loading {m_name} from: {ckpt_path.name}")
        model = models.build_model(m_name, num_classes=NUM_CLASSES, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        loaded_models[m_name] = model

    # Transform
    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Conditions to run
    if full_eval:
        conditions = ["clean"] + [
            f"{family}_{sev}" for family in ["blur", "dark", "bright", "jpeg"] for sev in [1, 2, 3]
        ]
    else:
        conditions = ["clean", "blur_3", "jpeg_3", "dark_3", "bright_3"]

    # Pre-allocate dictionary: logits[m_name][condition] = (N, 5)
    model_logits = {m_name: {c: [] for c in conditions} for m_name in model_names}
    all_targets = []

    batch_size = 64
    dataset = FastImageLoader(image_paths, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda b: b)

    start_eval = time.time()
    print("\nRunning inference across all 13 conditions on all 13,836 field images...")
    
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Batch Progress")):
            _, b_targets = zip(*batch)
            all_targets.extend(b_targets)

            # For each condition, transform and predict
            for cond in conditions:
                tensor_batch, _ = collate_and_corrupt(batch, cond, norm_transform)
                tensor_batch = tensor_batch.to(device)

                with torch.amp.autocast("cuda"):
                    for m_name in model_names:
                        out = loaded_models[m_name](tensor_batch)
                        model_logits[m_name][cond].append(out.cpu().numpy())

    total_time = time.time() - start_eval
    print(f"\nCompleted inference over {num_images} images across 13 conditions in {total_time:.2f}s (~{total_time/60:.2f} min)!")

    # Concatenate all logits
    targets_np = np.array(all_targets)
    for m_name in model_names:
        for cond in conditions:
            model_logits[m_name][cond] = np.concatenate(model_logits[m_name][cond], axis=0)

    # Save empirical logits to disk so they are never lost
    logits_save_path = config.OUTPUTS_DIR / "experiments" / "empirical_logits.npz"
    logits_save_path.parent.mkdir(parents=True, exist_ok=True)
    logits_dict = {"targets": targets_np}
    for m in model_names:
        for c in conditions:
            logits_dict[f"{m}_{c}"] = model_logits[m][c]
    np.savez_compressed(logits_save_path, **logits_dict)
    print(f"Saved all raw empirical logits to: {logits_save_path}")

    plot_dir = config.OUTPUTS_DIR / "plots" / "report"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # GENERATE FIGURE 5: Accuracy vs Coverage under 4 corruptions
    # =========================================================
    print("\nGenerating Figure 5: Accuracy on answered images vs coverage...")
    fig5_conditions = ["blur_3", "jpeg_3", "dark_3", "bright_3"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)

    coverages = np.linspace(0.02, 1.0, 100)

    for ax_idx, cond in enumerate(fig5_conditions):
        ax = axes[ax_idx]
        for m_name in model_names:
            logits = model_logits[m_name][cond]
            probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
            confidences = np.max(probs, axis=1)
            correctness = (np.argmax(probs, axis=1) == targets_np).astype(float)

            # Sort descending by confidence
            sorted_idx = np.argsort(-confidences)
            sorted_corr = correctness[sorted_idx]

            accs = []
            for cov in coverages:
                k = max(1, int(cov * num_images))
                accs.append(np.mean(sorted_corr[:k]))

            ax.plot(coverages, accs, label=m_name, color=MODEL_COLORS[m_name], linewidth=1.5)

        ax.set_title(cond, fontsize=11, fontweight="bold")
        ax.set_xlabel("coverage", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.35, 1.02)
        if ax_idx == 0:
            ax.set_ylabel("accuracy on answered images", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig5_path = plot_dir / "fig5_coverage_corruptions.png"
    fig.savefig(fig5_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 5: {fig5_path}")

    # =========================================================
    # GENERATE FIGURE 6: Reliability Diagrams Pooled Over Conditions
    # =========================================================
    print("\nGenerating Figure 6: Reliability diagrams pooled over all conditions...")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), sharey=True)

    num_bins = 10
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)

    for ax_idx, m_name in enumerate(model_names):
        ax = axes[ax_idx]

        # Pool logits and targets across conditions
        pooled_logits_list = [model_logits[m_name][c] for c in conditions]
        pooled_logits = np.concatenate(pooled_logits_list, axis=0)
        pooled_targets = np.tile(targets_np, len(conditions))

        # Temperature Scaling on pooled corruptions (T_mixed)
        scaler = TemperatureScaler()
        t_logits = torch.tensor(pooled_logits, dtype=torch.float32)
        t_targets = torch.tensor(pooled_targets, dtype=torch.long)
        scaler.fit(t_logits, t_targets, max_iter=80)
        t_mixed = float(scaler.temperature.detach().item())
        scaled_logits = (t_logits / scaler.temperature.clamp(min=1e-3)).detach().numpy()

        raw_probs = torch.softmax(t_logits, dim=1).numpy()
        cal_probs = torch.softmax(torch.tensor(scaled_logits), dim=1).numpy()

        raw_confs = np.max(raw_probs, axis=1)
        raw_preds = np.argmax(raw_probs, axis=1)
        raw_accs = (raw_preds == pooled_targets).astype(float)

        cal_confs = np.max(cal_probs, axis=1)
        cal_preds = np.argmax(cal_probs, axis=1)
        cal_accs = (cal_preds == pooled_targets).astype(float)

        # Binning for raw
        raw_bin_confs, raw_bin_accs = [], []
        cal_bin_confs, cal_bin_accs = [], []

        for b in range(num_bins):
            lo, hi = bin_boundaries[b], bin_boundaries[b+1]

            in_raw = (raw_confs > lo) & (raw_confs <= hi)
            if np.sum(in_raw) > 50:
                raw_bin_confs.append(np.mean(raw_confs[in_raw]))
                raw_bin_accs.append(np.mean(raw_accs[in_raw]))

            in_cal = (cal_confs > lo) & (cal_confs <= hi)
            if np.sum(in_cal) > 50:
                cal_bin_confs.append(np.mean(cal_confs[in_cal]))
                cal_bin_accs.append(np.mean(cal_accs[in_cal]))

        # Plot ideal diagonal
        ax.plot([0, 1], [0, 1], "k--", label="ideal", alpha=0.7)

        # Plot raw and calibrated
        ax.plot(raw_bin_confs, raw_bin_accs, "o-", color="#1f77b4", label="raw", linewidth=1.5, markersize=5)
        ax.plot(cal_bin_confs, cal_bin_accs, "o-", color="#ff7f0e", label="T_mixed", linewidth=1.5, markersize=5)

        ax.set_title(f"{m_name} (pooled corruptions)", fontsize=10, fontweight="bold")
        ax.set_xlabel("confidence", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        if ax_idx == 0:
            ax.set_ylabel("accuracy", fontsize=10)
            ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig6_path = plot_dir / "fig6_reliability_diagrams.png"
    fig.savefig(fig6_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 6: {fig6_path}")

    # =========================================================
    # GENERATE FIGURE 3: Confusion matrices under blur_3 and jpeg_3
    # =========================================================
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    cm_conditions = ["blur_3", "jpeg_3"]

    for row_idx, cond in enumerate(cm_conditions):
        for col_idx, m_name in enumerate(model_names):
            ax = axes[row_idx, col_idx]
            logits = model_logits[m_name][cond]
            preds = np.argmax(logits, axis=1)
            cm = confusion_matrix(targets_np, preds, labels=list(range(NUM_CLASSES)))

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                        xticklabels=[c[:6] for c in CLASSES],
                        yticklabels=[c[:6] for c in CLASSES] if col_idx == 0 else False)

            if row_idx == 0:
                ax.set_title(f"{m_name}\n({cond})", fontsize=10, fontweight="bold")
            else:
                ax.set_title(f"({cond})", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"True ({cond})", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig3_path = plot_dir / "fig3_confusion_matrices_corruptions.png"
    fig.savefig(fig3_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3: {fig3_path}")

    print("\n=======================================================")
    print("   ALL EMPIRICAL FIGURES GENERATED SUCCESSFULLY!")
    print("=======================================================\n")

    # Recompile PDF
    try:
        from myDoc.build_pdf_report import build_pdf
        print("Re-compiling publication PDF report...")
        build_pdf()
    except Exception as e:
        print(f"[Notice] PDF re-compilation can be run via: python myDoc/build_pdf_report.py (Error: {e})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Empirical Figures from Trained Models & 13,836 Images")
    parser.add_argument("--full", action="store_true", help="Evaluate all 13 conditions (takes ~30-35 mins)")
    parser.add_argument("--sample", type=int, default=None, help="Evaluate on a random sample of N images (e.g. 2000 for fast test)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference (default: 64)")
    args = parser.parse_args()

    run_empirical_eval(full_eval=args.full, sample_limit=args.sample, batch_size=args.batch_size)
