"""
End-to-End Scientific Experiment Runner: Robustness, Calibration & Explainability
Multi-Class Coconut Tree Disease Detection & Classification

Executes the full 4-stage empirical benchmark matching publication standards:
1. Clean Test Set Baseline (Macro-F1 with 95% Bootstrap CI, ECE, Brier Score, Clean Confusion Matrix)
2. Corruption Robustness (12 conditions: Blur, Dark, Bright, JPEG at 3 severities; Sink-class shift)
3. Probabilistic Calibration & Rejection Curves (Temperature Scaling, Accuracy at 100% to 50% Coverage)
4. Grad-CAM Stability, Cross-Model Agreement & Flip-Consistency Failure Detection AUROC

Saves all structured results to: outputs/experiments/experiment_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Add root directory to python path for config and models imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
import models
from corruptions import CorruptedDataset, CORRUPTIONS, ALL_CONDITIONS
from metrics_engine import (
    compute_macro_f1_with_bootstrap_ci,
    compute_ece,
    compute_brier_score,
    compute_risk_coverage,
    compute_error_detection_auroc,
    TemperatureScaler,
)
from cam_engine import (
    GradCAM,
    get_target_layer,
    compute_pearson_correlation,
    compute_ssim,
    compute_top_k_overlap,
    compute_flip_consistency,
)


class SimpleImageDataset(Dataset):
    """Loads image paths and target labels."""
    def __init__(self, image_paths: List[Path], targets: List[int]):
        self.image_paths = image_paths
        self.targets = targets

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        target = self.targets[idx]
        return img, target


def load_model_checkpoint(model_name: str, checkpoint_path: Path, device: torch.device):
    """Initializes model architecture and loads trained checkpoint weights."""
    print(f"Loading {model_name} from: {checkpoint_path.name}")
    model = models.build_model(model_name, num_classes=len(config.CLASSES), pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle state dict packaging
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


from tqdm import tqdm


def get_inference_outputs(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    desc: str = "Inference"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs inference and returns raw logits, probabilities, predictions, and targets with tqdm progress."""
    model.eval()
    all_logits, all_targets = [], []

    pbar = tqdm(dataloader, desc=desc, leave=False)
    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.numpy())

    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)

    return logits, probs, preds, targets


def run_experiment_pipeline(
    data_dir: Path = None,
    models_to_test: List[str] = None,
    cam_sample_size: int = 150,
    device_str: str = "cuda"
):
    start_time = time.time()
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"\n=======================================================")
    print(f"   COCONUT DISEASE EXPERIMENTAL BENCHMARK SUITE")
    print(f"   Execution Device: {device}")
    print(f"=======================================================\n")

    # 1. Resolve Dataset
    test_transform = transforms.Compose([
        transforms.Resize((int(config.IMAGE_SIZE[0] * 1.125), int(config.IMAGE_SIZE[1] * 1.125))),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_paths, targets = [], []
    if data_dir is not None and data_dir.exists():
        print(f"Scanning evaluation dataset from: {data_dir}")
        # 1. Check folder-based structure: data_dir / class_name
        for class_idx, class_name in enumerate(config.CLASSES):
            class_folder = data_dir / class_name
            if class_folder.exists() and class_folder.is_dir():
                files = [p for p in class_folder.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
                image_paths.extend(files)
                targets.extend([class_idx] * len(files))

        # 2. If no folder structure matched, search recursively by filename prefixes (e.g. Roboflow dumps)
        if len(image_paths) == 0:
            prefix_mapping = [
                ("BudRootDropping", 0),
                ("BudRot", 1),
                ("GrayLeafSpot", 2),
                ("LeafRot", 3),
                ("StemBleeding", 4),
            ]
            all_imgs = [p for p in data_dir.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            print(f"Scanning {len(all_imgs)} images recursively for class filename prefixes...")
            for img_path in all_imgs:
                name = img_path.name
                for prefix, label_idx in prefix_mapping:
                    if name.startswith(prefix):
                        image_paths.append(img_path)
                        targets.append(label_idx)
                        break

    else:
        test_csv = config.SPLITS_DIR / "test.csv"
        if not test_csv.exists():
            raise FileNotFoundError(f"Neither custom data_dir nor {test_csv} could be found!")
        print(f"Loading standard held-out test split from: {test_csv}")
        df_test = pd.read_csv(test_csv)
        image_paths = [Path(p) for p in df_test["image_path"].tolist()]
        targets = df_test["label"].tolist()

    print(f"Total Test Images Loaded: {len(image_paths)}")
    base_dataset = SimpleImageDataset(image_paths, targets)

    # 2. Candidate Models Selection
    if models_to_test is None:
        models_to_test = ["mobilenet_v2", "resnet50", "efficientnet_b0", "densenet121"]

    loaded_models = {}
    for m_name in models_to_test:
        ckpt_path = config.CHECKPOINTS_DIR / f"best_{m_name}.pth"
        if not ckpt_path.exists():
            print(f"[Warning] Checkpoint not found for {m_name} at {ckpt_path}. Skipping.")
            continue
        loaded_models[m_name] = load_model_checkpoint(m_name, ckpt_path, device)

    if not loaded_models:
        raise RuntimeError("No model checkpoints could be loaded. Ensure training has produced best_*.pth.")

    results_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_test_images": len(image_paths),
            "device": str(device),
            "classes": config.CLASSES,
            "models": list(loaded_models.keys()),
        },
        "clean_benchmarks": {},
        "corruptions": {},
        "prediction_distributions": {},
        "calibration_and_rejection": {},
        "grad_cam_stability": {},
        "cross_model_agreement": {},
        "flip_consistency_auroc": {},
    }

    # =========================================================
    # Phase 1: Clean Baseline Benchmark
    # =========================================================
    print("\n--- Phase 1: Clean Baseline Evaluation ---")
    clean_dataset = CorruptedDataset(base_dataset, "clean", severity=1, transform=test_transform)
    clean_loader = DataLoader(clean_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    clean_outputs = {}
    for m_name, model in loaded_models.items():
        logits, probs, preds, y_true = get_inference_outputs(
            model, clean_loader, device, desc=f"Evaluating Clean [{m_name}]"
        )
        clean_outputs[m_name] = {"logits": logits, "probs": probs, "preds": preds, "targets": y_true}

        acc = float(np.mean(preds == y_true))
        f1_dict = compute_macro_f1_with_bootstrap_ci(y_true, preds, num_bootstraps=500)
        ece = compute_ece(probs, y_true)
        brier = compute_brier_score(probs, y_true)
        num_errors = int(np.sum(preds != y_true))

        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, preds, labels=list(range(len(config.CLASSES)))).tolist()

        results_data["clean_benchmarks"][m_name] = {
            "accuracy": round(acc, 4),
            "macro_f1": f1_dict["macro_f1"],
            "ci_lower": f1_dict["ci_lower"],
            "ci_upper": f1_dict["ci_upper"],
            "ece": ece,
            "brier": brier,
            "errors": num_errors,
            "confusion_matrix": cm,
        }
        print(f"[{m_name}] Acc: {acc*100:.2f}% | Macro F1: {f1_dict['macro_f1']:.4f} [95% CI: {f1_dict['ci_lower']}-{f1_dict['ci_upper']}] | ECE: {ece:.4f} | Errors: {num_errors}")

    # =========================================================
    # Phase 2: Corruption Suite (12 conditions)
    # =========================================================
    print("\n--- Phase 2: Corruption Robustness Suite ---")
    corruption_conditions = [
        (family, sev) for family in ["blur", "dark", "bright", "jpeg"] for sev in [1, 2, 3]
    ]

    for m_name, model in loaded_models.items():
        results_data["corruptions"][m_name] = {"clean": results_data["clean_benchmarks"][m_name]["macro_f1"]}
        results_data["prediction_distributions"][m_name] = {}

    for family, sev in tqdm(corruption_conditions, desc="Corruptions Benchmark"):
        cond_name = f"{family}_{sev}"
        corr_dataset = CorruptedDataset(base_dataset, family, severity=sev, transform=test_transform)
        corr_loader = DataLoader(corr_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

        for m_name, model in loaded_models.items():
            _, probs, preds, y_true = get_inference_outputs(
                model, corr_loader, device, desc=f"[{m_name}] {cond_name}"
            )
            f1 = compute_macro_f1_with_bootstrap_ci(y_true, preds, num_bootstraps=100)["macro_f1"]
            results_data["corruptions"][m_name][cond_name] = f1

            # Distribution of predicted classes (identifies sink classes under blur)
            pred_counts = np.bincount(preds, minlength=len(config.CLASSES))
            pred_share = (pred_counts / len(preds)).round(4).tolist()
            results_data["prediction_distributions"][m_name][cond_name] = pred_share


    # Compute mean over 12 corruptions
    for m_name in loaded_models.keys():
        corr_vals = [results_data["corruptions"][m_name][f"{f}_{s}"] for f, s in corruption_conditions]
        results_data["corruptions"][m_name]["mean_corruptions"] = round(float(np.mean(corr_vals)), 4)

    # =========================================================
    # Phase 3: Calibration & Uncertainty Rejection
    # =========================================================
    print("\n--- Phase 3: Calibration & Selective Classification ---")
    for m_name in loaded_models.keys():
        clean_logits = torch.tensor(clean_outputs[m_name]["logits"])
        clean_targets = torch.tensor(clean_outputs[m_name]["targets"])

        # Temperature Scaling
        scaler = TemperatureScaler()
        opt_t = scaler.fit(clean_logits, clean_targets)

        # Scaled probabilities
        scaled_logits = scaler(clean_logits)
        scaled_probs = F.softmax(scaled_logits, dim=1).detach().numpy()

        raw_ece = results_data["clean_benchmarks"][m_name]["ece"]
        scaled_ece = compute_ece(scaled_probs, clean_outputs[m_name]["targets"])
        coverage_curve = compute_risk_coverage(clean_outputs[m_name]["probs"], clean_outputs[m_name]["targets"])

        results_data["calibration_and_rejection"][m_name] = {
            "optimal_temperature": opt_t,
            "raw_ece": raw_ece,
            "calibrated_ece": scaled_ece,
            "risk_coverage": coverage_curve,
        }
        print(f"[{m_name}] Temperature T: {opt_t:.2f} | Raw ECE: {raw_ece:.4f} -> Calibrated ECE: {scaled_ece:.4f}")

    # =========================================================
    # Phase 4: Grad-CAM Explainability & Failure Detection
    # =========================================================
    print("\n--- Phase 4: Grad-CAM Explainability & Heatmap Stability ---")
    sample_indices = np.random.default_rng(42).choice(len(base_dataset), size=min(cam_sample_size, len(base_dataset)), replace=False)
    print(f"Computing Grad-CAM across {len(sample_indices)} sample test images...")

    # Initialize GradCAM objects
    grad_cams = {}
    for m_name, model in loaded_models.items():
        layer = get_target_layer(model, m_name)
        grad_cams[m_name] = GradCAM(model, layer)

    cam_test_conditions = [("blur", 2), ("blur", 3), ("dark", 2), ("dark", 3), ("bright", 2), ("bright", 3), ("jpeg", 2), ("jpeg", 3)]
    cam_stability_records = {m_name: {f"{f}_{s}": [] for f, s in cam_test_conditions} for m_name in loaded_models.keys()}
    clean_cams = {m_name: [] for m_name in loaded_models.keys()}
    flip_consistency_scores = {m_name: [] for m_name in loaded_models.keys()}

    # Single-image transform (no batch dimension)
    single_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for idx in tqdm(sample_indices, desc="Computing Grad-CAM Heatmaps"):
        raw_pil, target_label = base_dataset[idx]
        clean_tensor = single_transform(raw_pil).unsqueeze(0).to(device)

        # 1. Clean CAMs
        sample_clean_cams = {}
        for m_name, g_cam in grad_cams.items():
            cam_map = g_cam.generate_heatmap(clean_tensor, target_class=target_label)
            sample_clean_cams[m_name] = cam_map
            clean_cams[m_name].append(cam_map)

            # Flip-consistency test
            flip_score = compute_flip_consistency(g_cam, clean_tensor, target_class=target_label)
            flip_consistency_scores[m_name].append(flip_score)

        # 2. Corrupted CAMs & Pearson Correlation
        from corruptions import corrupt_image
        for fam, sev in cam_test_conditions:
            cond_key = f"{fam}_{sev}"
            corrupted_pil = corrupt_image(raw_pil, fam, sev)
            corr_tensor = single_transform(corrupted_pil).unsqueeze(0).to(device)

            for m_name, g_cam in grad_cams.items():
                cam_corr = g_cam.generate_heatmap(corr_tensor, target_class=target_label)
                p_corr = compute_pearson_correlation(sample_clean_cams[m_name], cam_corr)
                cam_stability_records[m_name][cond_key].append(p_corr)

    # Average Pearson stability across conditions
    for m_name in loaded_models.keys():
        results_data["grad_cam_stability"][m_name] = {}
        for cond_key, scores in cam_stability_records[m_name].items():
            results_data["grad_cam_stability"][m_name][cond_key] = round(float(np.mean(scores)), 4)
        results_data["grad_cam_stability"][m_name]["mean_stability"] = round(float(np.mean(list(results_data["grad_cam_stability"][m_name].values()))), 4)

    # Clean up hooks
    for g_cam in grad_cams.values():
        g_cam.remove_hooks()

    # Cross-Model Agreement (on clean images)
    print("Computing Cross-Model Heatmap Agreement...")
    model_list = list(loaded_models.keys())
    for i in range(len(model_list)):
        for j in range(i + 1, len(model_list)):
            m1, m2 = model_list[i], model_list[j]
            pair_key = f"{m1}_vs_{m2}"
            pearsons, ious = [], []
            for k in range(len(sample_indices)):
                pearsons.append(compute_pearson_correlation(clean_cams[m1][k], clean_cams[m2][k]))
                ious.append(compute_top_k_overlap(clean_cams[m1][k], clean_cams[m2][k], top_fraction=0.20))
            results_data["cross_model_agreement"][pair_key] = {
                "mean_pearson": round(float(np.mean(pearsons)), 4),
                "mean_top20_overlap": round(float(np.mean(ious)), 4),
            }

    # Failure Detection via Flip Consistency
    for m_name in loaded_models.keys():
        sample_preds = clean_outputs[m_name]["preds"][sample_indices]
        sample_targets = clean_outputs[m_name]["targets"][sample_indices]
        sample_confs = np.max(clean_outputs[m_name]["probs"][sample_indices], axis=1)
        instability_scores = 1.0 - np.array(flip_consistency_scores[m_name])

        conf_auroc = compute_error_detection_auroc(sample_confs, sample_targets, sample_preds)
        cam_auroc = compute_error_detection_auroc(1.0 - instability_scores, sample_targets, sample_preds)

        results_data["flip_consistency_auroc"][m_name] = {
            "confidence_auroc": conf_auroc,
            "cam_instability_auroc": cam_auroc,
        }

    # =========================================================
    # Phase 5: Save Structured Report JSON
    # =========================================================
    out_dir = config.OUTPUTS_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "experiment_results.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)

    total_time = (time.time() - start_time) / 60.0
    print(f"\n=======================================================")
    print(f"   Benchmark Completed in {total_time:.2f} minutes!")
    print(f"   Saved raw experiment data to: {json_path}")
    print(f"=======================================================\n")
    return json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Scientific Experiment Benchmark")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to external image evaluation dataset")
    parser.add_argument("--models", nargs="+", default=["mobilenet_v2", "resnet50", "efficientnet_b0", "densenet121"])
    parser.add_argument("--cam_samples", type=int, default=150, help="Number of samples for Grad-CAM stability testing")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()
    data_dir_path = Path(args.data_dir) if args.data_dir else None

    run_experiment_pipeline(
        data_dir=data_dir_path,
        models_to_test=args.models,
        cam_sample_size=args.cam_samples,
        device_str=args.device,
    )
