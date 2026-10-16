"""
Standardized Training Pipeline with Imbalance Mitigation
Multi-Class Coconut Tree Disease Detection & Classification

Features:
- Dynamically selects Candidate Model (ResNet50, MobileNetV2, EfficientNet-B0, DenseNet121, VGG16)
- Implements Cost-Sensitive Loss (Weighted Cross-Entropy or Focal Loss)
- Uses WeightedRandomSampler to prevent minority class starvation
- Checkpoints best model strictly on Macro F1-Score (unbiased by majority class size)
- Saves training curves (loss, accuracy, macro-F1)
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

import config
import dataset
import loss
import metrics
import models

# Optional tqdm for clean progress bars, with fallback
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Executes one training epoch."""
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    pbar = tqdm(dataloader, desc="Training")
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        batch_loss = criterion(outputs, targets)
        batch_loss.backward()
        optimizer.step()

        running_loss += batch_loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_metrics = metrics.compute_metrics(
        np.array(all_targets),
        np.array(all_preds),
        num_classes=config.NUM_CLASSES,
        class_names=config.CLASSES,
    )
    return epoch_loss, epoch_metrics


@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device):
    """Evaluates the model on validation or test split."""
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        batch_loss = criterion(outputs, targets)

        running_loss += batch_loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_metrics = metrics.compute_metrics(
        np.array(all_targets),
        np.array(all_preds),
        num_classes=config.NUM_CLASSES,
        class_names=config.CLASSES,
    )
    return epoch_loss, epoch_metrics


def plot_training_curves(history: dict, save_path: Path):
    """Plots and saves loss, accuracy, and Macro F1 curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r--s", label="Val Loss")
    axes[0].set_title("Cross-Entropy / Focal Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], "g--s", label="Val Accuracy")
    axes[1].set_title("Overall Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Macro F1-Score (The Key Metric for Imbalance)
    axes[2].plot(epochs, history["train_macro_f1"], "b-o", label="Train Macro F1")
    axes[2].plot(epochs, history["val_macro_f1"], "m--^", label="Val Macro F1 (Key)")
    axes[2].set_title("Macro F1-Score (Balanced Performance)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {save_path}")


from typing import Optional

def run_training(
    model_name: str = "mobilenet_v2",
    loss_type: str = config.LOSS_TYPE,
    epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    use_balanced_sampler: bool = config.USE_BALANCED_SAMPLER,
    force_resplit: bool = False,
    samples_per_class: Optional[int] = config.SAMPLES_PER_CLASS,
    freeze_backbone: bool = False,
    dropout: Optional[float] = None,
    weight_decay: float = config.WEIGHT_DECAY,
    patience: int = 7,
    warmup_epochs: int = config.WARMUP_EPOCHS,
    label_smoothing: float = config.LABEL_SMOOTHING,
):
    print("=" * 70)
    print("=== COCONUT TREE DISEASE CLASSIFICATION & DETECTION ===")
    print(f"Model: {model_name} | Loss: {loss_type} (Smoothing: {label_smoothing})")
    print(f"Device: {config.DEVICE} | Image Size: {config.IMAGE_SIZE} | Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"Balanced Sampler: {use_balanced_sampler} | Warmup Epochs: {warmup_epochs} | Early Stop Patience: {patience}")
    if samples_per_class:
        print(f"Dataset Balancing: Active (Capped at {samples_per_class} images/class)")
    else:
        print("Dataset Balancing: Full Dataset (All 5,798 images utilizing WeightedRandomSampler)")
    print("=" * 70)

    # 1. DataLoaders & Class Imbalance Profiling
    train_loader, val_loader, test_loader, train_counts = dataset.get_dataloaders(
        batch_size=batch_size,
        use_balanced_sampler=use_balanced_sampler,
        force_resplit=force_resplit,
        max_samples_per_class=samples_per_class,
    )
    print("\nTraining Split Class Counts:")
    for cls_name, count in train_counts.items():
        print(f"  - {cls_name:20s}: {count:5d} images")

    # 2. Compute Loss with Class Weighting & Label Smoothing
    class_weights = loss.compute_class_weights(train_counts, method="inverse_freq")
    criterion = loss.get_loss_function(
        loss_type,
        class_weights=class_weights,
        device=config.DEVICE,
        label_smoothing=label_smoothing,
    )
    print(f"\nLoss configured as: {loss_type.upper()} with label smoothing={label_smoothing}")
    for cls_name, w in zip(config.CLASSES, class_weights):
        print(f"  - Penalty weight for {cls_name:20s}: {w.item():.3f}")

    # 3. Model Architecture
    model = models.build_model(
        model_name=model_name,
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )
    model = model.to(config.DEVICE)
    model_info = models.get_model_summary(model)
    print(f"\nArchitecture: {model_name}")
    print(f"  Total Parameters: {model_info['total_params']:,} | Trainable: {model_info['trainable_params']:,} | Model Size: {model_info['size_mb']} MB")

    # 4. Two-Stage Optimizer & LR Scheduler Setup
    use_two_stage = (warmup_epochs > 0 and not freeze_backbone)
    if use_two_stage:
        print(f"\n[Two-Stage Fine-Tuning Active]:")
        print(f"  - Stage 1: Warmup classification head for {warmup_epochs} epochs (Backbone frozen)")
        print(f"  - Stage 2: Full backbone fine-tuning (Differential LR: Backbone {learning_rate * 0.1:.1e}, Head {learning_rate:.1e})")

        # Freeze backbone initially
        if hasattr(model, "features"):
            for p in model.features.parameters():
                p.requires_grad = False
        elif hasattr(model, "fc"):
            for n, p in model.named_parameters():
                if not n.startswith("fc"):
                    p.requires_grad = False

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=warmup_epochs, eta_min=1e-5)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 5. Training Loop
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_macro_f1": [], "val_macro_f1": [],
    }

    best_val_macro_f1 = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_file = config.CHECKPOINT_DIR / f"best_{model_name}.pth"

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        # Stage Transition: Unfreeze backbone after warmup
        if use_two_stage and epoch == warmup_epochs + 1:
            print(f"\n[Stage 2 Transition] Unfreezing backbone with Differential Learning Rates...")
            for p in model.parameters():
                p.requires_grad = True

            if hasattr(model, "features"):
                param_groups = [
                    {"params": model.features.parameters(), "lr": learning_rate * 0.1},
                    {"params": model.classifier.parameters(), "lr": learning_rate},
                ]
            elif hasattr(model, "fc"):
                backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
                param_groups = [
                    {"params": backbone_params, "lr": learning_rate * 0.1},
                    {"params": model.fc.parameters(), "lr": learning_rate},
                ]
            else:
                param_groups = [{"params": model.parameters(), "lr": learning_rate}]

            optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
            remaining_epochs = max(epochs - warmup_epochs, 1)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=1e-6)
            print(f"  Backbone LR: {learning_rate * 0.1:.1e} | Head LR: {learning_rate:.1e} | Remaining Epochs: {remaining_epochs}")

        stage_name = "Stage 1 (Head Warmup)" if (use_two_stage and epoch <= warmup_epochs) else "Stage 2 (Full Fine-Tune)"
        print(f"\n--- Epoch {epoch}/{epochs} [{stage_name}] ---")
        train_loss, train_met = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_met = evaluate_epoch(model, val_loader, criterion, config.DEVICE)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_met["accuracy"])
        history["val_acc"].append(val_met["accuracy"])
        history["train_macro_f1"].append(train_met["macro_f1"])
        history["val_macro_f1"].append(val_met["macro_f1"])

        print(
            f"Train Loss: {train_loss:.4f} | Acc: {train_met['accuracy']*100:.2f}% | Macro F1: {train_met['macro_f1']:.4f}\n"
            f"Val   Loss: {val_loss:.4f} | Acc: {val_met['accuracy']*100:.2f}% | Macro F1: {val_met['macro_f1']:.4f}"
        )

        # Monitor Macro F1-Score for Best Model Selection
        improved = False
        if val_met["macro_f1"] > best_val_macro_f1:
            improved = True
        elif val_met["macro_f1"] == best_val_macro_f1 and val_loss < best_val_loss:
            improved = True

        if improved:
            best_val_macro_f1 = val_met["macro_f1"]
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_macro_f1": best_val_macro_f1,
                "val_accuracy": val_met["accuracy"],
                "class_names": config.CLASSES,
            }, checkpoint_file)
            print(f"[BEST] Saved new best checkpoint to {checkpoint_file.name} (Val Macro F1: {best_val_macro_f1:.4f})")
        else:
            if not use_two_stage or epoch > warmup_epochs:
                patience_counter += 1

        if patience > 0 and patience_counter >= patience:
            print(f"\n[Early Stopping] Validation performance did not improve for {patience} consecutive epochs. Stopping to prevent overfitting.")
            break

    elapsed_min = (time.time() - start_time) / 60.0
    print(f"\nTraining completed in {elapsed_min:.2f} minutes.")
    print(f"Peak Validation Macro F1: {best_val_macro_f1:.4f}")

    # Save History and Plot
    history_file = config.METRICS_DIR / f"history_{model_name}.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    curves_file = config.PLOTS_DIR / f"curves_{model_name}.png"
    plot_training_curves(history, curves_file)

    return checkpoint_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Coconut Disease Classifier")
    parser.add_argument("--model", type=str, default="mobilenet_v2", choices=config.CANDIDATE_MODELS + ["vgg16"],
                        help="Candidate model backbone")
    parser.add_argument("--loss", type=str, default=config.LOSS_TYPE, choices=["standard_ce", "weighted_ce", "focal"],
                        help="Loss function")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY,
                        help="L2 weight decay regularization (default: 1e-4)")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Dropout rate in classification head (e.g. 0.3, 0.4, 0.5)")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone convolutional layers and only train classification head")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience in epochs (default: 7, set 0 to disable)")
    parser.add_argument("--no_sampler", action="store_true", help="Disable WeightedRandomSampler")
    parser.add_argument("--resplit", action="store_true",
                        help="Force regenerating train/val/test splits before training")
    parser.add_argument("--samples_per_class", type=int, default=None,
                        help="Max images per class (default: None for all 5,798 images)")
    parser.add_argument("--warmup_epochs", type=int, default=config.WARMUP_EPOCHS,
                        help="Head warmup epochs before unfreezing backbone")
    parser.add_argument("--label_smoothing", type=float, default=config.LABEL_SMOOTHING,
                        help="Label smoothing factor (default: 0.1)")

    args = parser.parse_args()

    run_training(
        model_name=args.model,
        loss_type=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        patience=args.patience,
        use_balanced_sampler=not args.no_sampler,
        force_resplit=args.resplit,
        samples_per_class=args.samples_per_class,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
    )