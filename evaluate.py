"""
Model Evaluation, Benchmark Comparison, and Confusion Matrix Generator
Multi-Class Coconut Tree Disease Detection & Classification

Outputs:
1. Per-class Precision, Recall, F1-Score (essential for assessing Bud Rot & Bud Root Dropping)
2. Confusion Matrix Heatmap (PNG)
3. Inference Latency (ms/img) and Throughput (FPS)
4. Comparative benchmark table across all candidate architectures (CSV & Markdown)
"""

import argparse
import json
import time
from pathlib import Path
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
import dataset
import metrics
import models


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: Path, title: str = "Confusion Matrix"):
    """Plots and saves an annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Disease Label",
        xlabel="Predicted Disease Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    # Annotate counts inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to: {save_path}")


@torch.no_grad()
def benchmark_latency(model: torch.nn.Module, device: str, input_size=None, runs: int = 100):
    """Measures single-image inference latency (ms) and throughput (FPS)."""
    if input_size is None:
        input_size = (1, 3, *config.IMAGE_SIZE)
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)

    # Warmup
    for _ in range(10):
        _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(runs):
        _ = model(dummy_input)
    if device == "cuda":
        torch.cuda.synchronize()

    total_time = time.time() - start_time
    avg_latency_ms = (total_time / runs) * 1000.0
    fps = runs / total_time
    return round(avg_latency_ms, 2), round(fps, 1)


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: Path, eval_remaining: bool = False):
    """
    Evaluates a single model checkpoint on either the test set or the remaining
    unseen out-of-sample images.
    """
    target_name = "Remaining Out-of-Sample Images" if eval_remaining else "Test Set"
    print("=" * 70)
    print(f"Evaluating Checkpoint: {checkpoint_path.name} on {target_name}")
    print("=" * 70)

    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model_name = checkpoint.get("model_name", checkpoint_path.stem.replace("best_", ""))

    # Build and load model weights
    model = models.build_model(model_name=model_name, num_classes=config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.DEVICE)
    model.eval()

    # Model complexity
    summary = models.get_model_summary(model)
    latency_ms, fps = benchmark_latency(model, config.DEVICE)

    # Load appropriate dataset
    if eval_remaining:
        eval_loader = dataset.get_remaining_loader(batch_size=config.BATCH_SIZE)
    else:
        _, _, eval_loader, _ = dataset.get_dataloaders(batch_size=config.BATCH_SIZE, use_balanced_sampler=False)

    total_eval_samples = len(eval_loader.dataset)
    print(f"Dataset Size: {total_eval_samples} images")

    all_preds, all_targets = [], []
    for images, targets in eval_loader:
        images = images.to(config.DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    eval_results = metrics.compute_metrics(
        y_true, y_pred,
        num_classes=config.NUM_CLASSES,
        class_names=config.CLASSES,
    )

    # Print Report
    split_label = "Remaining Images" if eval_remaining else "Test Set"
    print(f"\nOverall {split_label} Accuracy: {eval_results['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:             {eval_results['macro_precision']:.4f}")
    print(f"Macro Recall:                {eval_results['macro_recall']:.4f}")
    print(f"Macro F1-Score:              {eval_results['macro_f1']:.4f}")
    print(f"Weighted F1-Score:           {eval_results['weighted_f1']:.4f}")
    print(f"Inference Latency:           {latency_ms} ms/image ({fps} FPS)")
    print(f"Model Parameters:            {summary['total_params']:,} ({summary['size_mb']} MB)\n")

    print(f"{'Disease Class':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 70)
    for cls_name, p in eval_results["per_class"].items():
        print(f"{cls_name:<22} | {p['precision']:<10.4f} | {p['recall']:<10.4f} | {p['f1_score']:<10.4f} | {p['support']:<8d}")

    # Plot Confusion Matrix
    prefix = "cm_remaining_" if eval_remaining else "cm_"
    title_suffix = " (Remaining Images)" if eval_remaining else " (Test Set)"
    cm_plot_path = config.PLOTS_DIR / f"{prefix}{model_name}.png"
    plot_confusion_matrix(eval_results["confusion_matrix"], config.CLASSES, cm_plot_path,
                          title=f"Confusion Matrix: {model_name}{title_suffix}")

    # Save detailed JSON metrics
    metrics_prefix = "eval_remaining_" if eval_remaining else "eval_"
    metrics_path = config.METRICS_DIR / f"{metrics_prefix}{model_name}.json"
    save_data = {
        "model_name": model_name,
        "evaluation_split": "remaining" if eval_remaining else "test",
        "sample_count": total_eval_samples,
        "total_params": summary["total_params"],
        "size_mb": summary["size_mb"],
        "latency_ms": latency_ms,
        "fps": fps,
        "accuracy": eval_results["accuracy"],
        "macro_precision": eval_results["macro_precision"],
        "macro_recall": eval_results["macro_recall"],
        "macro_f1": eval_results["macro_f1"],
        "weighted_f1": eval_results["weighted_f1"],
        "per_class": eval_results["per_class"],
    }
    with open(metrics_path, "w") as f:
        json.dump(save_data, f, indent=2)

    return save_data


@torch.no_grad()
def evaluate_ensemble(eval_remaining: bool = False, weights: dict = None):
    """
    Evaluates a Soft-Voting Ensemble combining all trained candidate models in
    outputs/checkpoints/ by averaging their softmax probabilities.
    """
    target_name = "Remaining Out-of-Sample Images" if eval_remaining else "Test Set"
    print("=" * 70)
    print(f"Evaluating Soft-Voting Ensemble on {target_name}")
    print("=" * 70)

    checkpoint_files = sorted(list(config.CHECKPOINT_DIR.glob("best_*.pth")))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {config.CHECKPOINT_DIR}")

    loaded_models = []
    model_names = []
    total_params = 0
    total_size_mb = 0.0

    for ckpt_path in checkpoint_files:
        checkpoint = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
        m_name = checkpoint.get("model_name", ckpt_path.stem.replace("best_", ""))
        m = models.build_model(model_name=m_name, num_classes=config.NUM_CLASSES, pretrained=False)
        m.load_state_dict(checkpoint["model_state_dict"])
        m = m.to(config.DEVICE)
        m.eval()
        loaded_models.append(m)
        model_names.append(m_name)
        s = models.get_model_summary(m)
        total_params += s["total_params"]
        total_size_mb += s["size_mb"]
        print(f"  Loaded candidate: {m_name} ({ckpt_path.name})")

    # Weight assignment: default 0.55 MobileNetV2 + 0.45 ResNet50
    if weights is None:
        if "mobilenet_v2" in model_names and "resnet50" in model_names and len(model_names) == 2:
            model_weights = [0.55 if name == "mobilenet_v2" else 0.45 for name in model_names]
        else:
            model_weights = [1.0 / len(loaded_models)] * len(loaded_models)
    else:
        model_weights = [weights.get(name, 1.0 / len(loaded_models)) for name in model_names]
        w_sum = sum(model_weights)
        model_weights = [w / w_sum for w in model_weights]

    print("\nEnsemble Weighting Breakdown:")
    for name, w in zip(model_names, model_weights):
        print(f"  - {name:<15}: {w * 100:.1f}%")

    if eval_remaining:
        eval_loader = dataset.get_remaining_loader(batch_size=config.BATCH_SIZE)
    else:
        _, _, eval_loader, _ = dataset.get_dataloaders(batch_size=config.BATCH_SIZE, use_balanced_sampler=False)

    total_eval_samples = len(eval_loader.dataset)
    print(f"\nDataset Size: {total_eval_samples} images")

    all_preds, all_targets = [], []
    start_time = time.time()

    for images, targets in eval_loader:
        images = images.to(config.DEVICE)
        ensemble_probs = torch.zeros((images.size(0), config.NUM_CLASSES), device=config.DEVICE)

        for m, w in zip(loaded_models, model_weights):
            logits = m(images)
            probs = torch.softmax(logits, dim=1)
            ensemble_probs += w * probs

        preds = torch.argmax(ensemble_probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())

    total_time = time.time() - start_time
    latency_ms = round((total_time / total_eval_samples) * 1000.0, 2)
    fps = round(total_eval_samples / total_time, 1)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    eval_results = metrics.compute_metrics(
        y_true, y_pred,
        num_classes=config.NUM_CLASSES,
        class_names=config.CLASSES,
    )

    split_label = "Remaining Images" if eval_remaining else "Test Set"
    print(f"\n=== SOFT-VOTING ENSEMBLE RESULTS ({split_label}) ===")
    print(f"Overall Accuracy:            {eval_results['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:             {eval_results['macro_precision']:.4f}")
    print(f"Macro Recall:                {eval_results['macro_recall']:.4f}")
    print(f"Macro F1-Score:              {eval_results['macro_f1']:.4f}")
    print(f"Weighted F1-Score:           {eval_results['weighted_f1']:.4f}")
    print(f"Pipeline Latency:            {latency_ms} ms/image ({fps} FPS)")
    print(f"Combined Parameters:         {total_params:,} ({total_size_mb:.2f} MB)\n")

    print(f"{'Disease Class':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 70)
    for cls_name, p in eval_results["per_class"].items():
        print(f"{cls_name:<22} | {p['precision']:<10.4f} | {p['recall']:<10.4f} | {p['f1_score']:<10.4f} | {p['support']:<8d}")

    # Plot Confusion Matrix
    prefix = "cm_remaining_" if eval_remaining else "cm_"
    title_suffix = " (Remaining Images)" if eval_remaining else " (Test Set)"
    cm_plot_path = config.PLOTS_DIR / f"{prefix}ensemble.png"
    plot_confusion_matrix(eval_results["confusion_matrix"], config.CLASSES, cm_plot_path,
                          title=f"Confusion Matrix: Soft-Voting Ensemble{title_suffix}")

    # Save detailed JSON metrics
    metrics_prefix = "eval_remaining_" if eval_remaining else "eval_"
    metrics_path = config.METRICS_DIR / f"{metrics_prefix}ensemble.json"
    save_data = {
        "model_name": "soft_voting_ensemble",
        "evaluation_split": "remaining" if eval_remaining else "test",
        "models_combined": model_names,
        "weights": dict(zip(model_names, model_weights)),
        "sample_count": total_eval_samples,
        "total_params": total_params,
        "size_mb": round(total_size_mb, 2),
        "latency_ms": latency_ms,
        "fps": fps,
        "accuracy": eval_results["accuracy"],
        "macro_precision": eval_results["macro_precision"],
        "macro_recall": eval_results["macro_recall"],
        "macro_f1": eval_results["macro_f1"],
        "weighted_f1": eval_results["weighted_f1"],
        "per_class": eval_results["per_class"],
    }
    with open(metrics_path, "w") as f:
        json.dump(save_data, f, indent=2)

    return save_data


def compare_all_models(eval_remaining: bool = False, include_ensemble: bool = True):
    """Gathers all evaluated models and generates a side-by-side comparison table."""
    results = []
    checkpoint_files = sorted(list(config.CHECKPOINT_DIR.glob("best_*.pth")))

    if not checkpoint_files:
        print(f"No checkpoint files found in {config.CHECKPOINT_DIR}. Train models first using train.py.")
        return

    for ckpt in checkpoint_files:
        res = evaluate_checkpoint(ckpt, eval_remaining=eval_remaining)
        results.append({
            "Model Architecture": res["model_name"],
            "Accuracy (%)": round(res["accuracy"] * 100, 2),
            "Macro F1": res["macro_f1"],
            "Weighted F1": res["weighted_f1"],
            "Macro Recall": res["macro_recall"],
            "Macro Precision": res["macro_precision"],
            "Params (M)": round(res["total_params"] / 1e6, 2),
            "Size (MB)": res["size_mb"],
            "Latency (ms)": res["latency_ms"],
            "FPS": res["fps"],
        })

    # Include Soft-Voting Ensemble if multiple models exist
    if include_ensemble and len(checkpoint_files) > 1:
        ens_res = evaluate_ensemble(eval_remaining=eval_remaining)
        results.append({
            "Model Architecture": "Soft-Voting Ensemble",
            "Accuracy (%)": round(ens_res["accuracy"] * 100, 2),
            "Macro F1": ens_res["macro_f1"],
            "Weighted F1": ens_res["weighted_f1"],
            "Macro Recall": ens_res["macro_recall"],
            "Macro Precision": ens_res["macro_precision"],
            "Params (M)": round(ens_res["total_params"] / 1e6, 2),
            "Size (MB)": ens_res["size_mb"],
            "Latency (ms)": ens_res["latency_ms"],
            "FPS": ens_res["fps"],
        })

    df = pd.DataFrame(results).sort_values(by="Macro F1", ascending=False)
    csv_filename = "model_comparison_remaining.csv" if eval_remaining else "model_comparison.csv"
    csv_path = config.METRICS_DIR / csv_filename
    df.to_csv(csv_path, index=False)

    split_title = "ON REMAINING OUT-OF-SAMPLE IMAGES" if eval_remaining else "ON TEST SPLIT"
    print("\n" + "=" * 80)
    print(f"=== FINAL COMPARATIVE BENCHMARK SUMMARY {split_title} (Sorted by Macro F1) ===")
    print("=" * 80)
    print(df.to_string(index=False))
    print(f"\nSaved comparison table to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Models and Soft-Voting Ensemble")
    parser.add_argument("--model", type=str, default=None, help="Name of model to evaluate (e.g. resnet50, mobilenet_v2)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to single model checkpoint .pth")
    parser.add_argument("--all", action="store_true", help="Benchmark all checkpoints in outputs/checkpoints/")
    parser.add_argument("--remaining", action="store_true", help="Evaluate models on remaining out-of-sample images (unseen, outside 470-sample splits)")
    parser.add_argument("--ensemble", action="store_true", help="Evaluate soft-voting ensemble of all available models")

    args = parser.parse_args()

    if args.ensemble:
        evaluate_ensemble(eval_remaining=args.remaining)
    elif args.model:
        ckpt_path = config.CHECKPOINT_DIR / f"best_{args.model}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Available: {[f.name for f in config.CHECKPOINT_DIR.glob('best_*.pth')]}")
        evaluate_checkpoint(ckpt_path, eval_remaining=args.remaining)
    elif args.checkpoint:
        evaluate_checkpoint(Path(args.checkpoint), eval_remaining=args.remaining)
    else:
        compare_all_models(eval_remaining=args.remaining)


