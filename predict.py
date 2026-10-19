"""
Single-Image & Batch Inference Script with Explainability
Multi-Class Coconut Tree Disease Detection & Classification

Usage:
  python predict.py --image sample.jpg
  python predict.py --image sample.jpg --explain
  python predict.py --dir path/to/images/ --output_csv predictions.csv
"""

import argparse
from pathlib import Path
from PIL import Image
import pandas as pd
import torch
import torch.nn.functional as F

from src import config, dataset, explainability, models


def load_model_from_checkpoint(checkpoint_path: Path):
    """Loads model architecture and pre-trained weights."""
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model_name = checkpoint.get("model_name", "resnet50")

    model = models.build_model(model_name, num_classes=config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.DEVICE)
    model.eval()
    return model, model_name


def predict_single_image(model, image_path: Path, topk: int = 3):
    """Predicts disease class and top-K probabilities for an image."""
    transform = dataset.get_transforms("val")
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=topk)
    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "class_name": config.CLASSES[idx.item()],
            "confidence": float(prob.item()),
            "class_index": int(idx.item()),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Inference for Coconut Tree Disease Detection")
    parser.add_argument("--image", type=str, default=None, help="Path to single image file")
    parser.add_argument("--dir", type=str, default=None, help="Directory containing images for batch inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--explain", action="store_true", help="Generate Grad-CAM symptom localization map")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Output CSV path for batch inference")

    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        candidates = list(config.CHECKPOINT_DIR.glob("best_*.pth"))
        if not candidates:
            print(f"Error: No model checkpoint found in {config.CHECKPOINT_DIR}. Train a model first with train.py.")
            return
        ckpt_path = candidates[0]

    print(f"Loading checkpoint: {ckpt_path.name}")
    model, model_name = load_model_from_checkpoint(ckpt_path)

    # 1. Single Image Inference
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Error: Image not found at {img_path}")
            return

        preds = predict_single_image(model, img_path)
        top = preds[0]

        print("\n" + "=" * 50)
        print(f"DIAGNOSIS RESULT FOR: {img_path.name}")
        print("=" * 50)
        print(f"Predicted Disease:  {top['class_name'].upper()}")
        print(f"Confidence:         {top['confidence'] * 100:.2f}%\n")
        print("Top Predictions:")
        for i, p in enumerate(preds, 1):
            print(f"  {i}. {p['class_name']:<22} : {p['confidence'] * 100:.2f}%")
        print("=" * 50)

        if args.explain:
            save_cam_path = config.PLOTS_DIR / f"gradcam_{img_path.stem}.png"
            explainability.explain_image(img_path, ckpt_path, save_cam_path)

    # 2. Batch Inference
    elif args.dir:
        dir_path = Path(args.dir)
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            image_files.extend(list(dir_path.glob(ext)))

        if not image_files:
            print(f"No image files found in {dir_path}")
            return

        print(f"Running inference on {len(image_files)} images...")
        rows = []
        for img_p in image_files:
            preds = predict_single_image(model, img_p, topk=1)
            rows.append({
                "image_name": img_p.name,
                "predicted_disease": preds[0]["class_name"],
                "confidence": round(preds[0]["confidence"], 4),
            })

        df = pd.DataFrame(rows)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved batch predictions to: {args.output_csv}")
    else:
        print("Please provide --image <path> or --dir <path>.")


if __name__ == "__main__":
    main()
