"""
Dataset Verification, Setup, and Stratified Split Utility
Multi-Class Coconut Tree Disease Detection & Classification

Usage:
  1. Set up directory folders for the 5 classes:
     python prepare_data.py --setup

  2. Verify downloaded images and perform Stratified Train/Val/Test Split:
     python prepare_data.py --split

  3. (Optional) Create small synthetic placeholder images to test the entire pipeline end-to-end:
     python prepare_data.py --mock
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config, dataset


def setup_class_directories(target_dir: Path = config.RAW_DATA_DIR):
    """Creates the standard class subfolders."""
    print(f"Creating class directories in {target_dir}...")
    for cls_name in config.CLASSES:
        p = target_dir / cls_name
        p.mkdir(parents=True, exist_ok=True)
        print(f"  [DIR] {p.resolve()}")
    print("\n[OK] Place your downloaded image files (.jpg / .png) into their corresponding class folders above.")


def verify_and_split(samples_per_class: Optional[int] = config.SAMPLES_PER_CLASS):
    """Audits the raw dataset and generates stratified CSV splits."""
    print("=" * 60)
    print("AUDITING COCONUT DISEASE DATASET")
    print("=" * 60)

    if not config.RAW_DATA_DIR.exists():
        print(f"Error: Directory {config.RAW_DATA_DIR} does not exist. Run with --setup first.")
        return

    # Count images per class
    counts = {}
    total_imgs = 0
    for cls_name in config.CLASSES:
        cls_dir = config.RAW_DATA_DIR / cls_name
        if not cls_dir.exists():
            # Try lowercase or underscore
            candidates = [
                config.RAW_DATA_DIR / cls_name.lower(),
                config.RAW_DATA_DIR / cls_name.lower().replace(" ", "_"),
                config.RAW_DATA_DIR / cls_name.replace(" ", "_"),
            ]
            cls_dir = next((c for c in candidates if c.exists()), None)

        if cls_dir is not None:
            seen = set()
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                for p in cls_dir.glob(ext):
                    seen.add(str(p.resolve()).lower())
            counts[cls_name] = len(seen)
            total_imgs += len(seen)
        else:
            counts[cls_name] = 0

    print(f"\nDiscovered {total_imgs} total images:")
    for cls_name, c in counts.items():
        pct = (c / total_imgs * 100) if total_imgs > 0 else 0
        expected = config.BENCHMARK_CLASS_COUNTS.get(cls_name, 0)
        target_info = f"-> Balanced to {min(c, samples_per_class)}" if samples_per_class else ""
        print(f"  - {cls_name:20s}: {c:5d} images ({pct:5.1f}%) | [Paper benchmark: {expected}] {target_info}")

    if total_imgs == 0:
        print("\n[WARNING] No images found in data/raw/. Please populate the folders with images, or run with --mock.")
        return

    if samples_per_class:
        print(f"\n[BALANCED MODE] Balancing active: capping each class at {samples_per_class} images (matching minority class).")
    else:
        print("\n[NATURAL MODE] Using all available images (natural imbalance).")

    print("\nPerforming Stratified 70% Train / 15% Val / 15% Test Split...")
    dataset.create_stratified_splits(
        raw_data_dir=config.RAW_DATA_DIR,
        output_dir=config.SPLIT_DATA_DIR,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        random_seed=config.RANDOM_SEED,
        max_samples_per_class=samples_per_class,
    )
    print("\n[OK] Stratified splits successfully generated in data/splits/")


def generate_mock_dataset(num_samples_per_class: int = 15):
    """
    Creates dummy synthetic images matching the 5 classes so you can test
    the entire training, evaluation, and explainability pipeline immediately.
    """
    print(f"Generating mock dataset ({num_samples_per_class} sample images per class)...")
    np.random.seed(config.RANDOM_SEED)

    for idx, cls_name in enumerate(config.CLASSES):
        cls_dir = config.RAW_DATA_DIR / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples_per_class):
            # Create a simple synthetic image with a distinct hue per class
            color = np.array([50 + idx * 40, 100 + (idx % 2) * 80, 70 + idx * 30], dtype=np.uint8)
            img_arr = np.ones((224, 224, 3), dtype=np.uint8) * color
            noise = np.random.randint(-25, 25, (224, 224, 3), dtype=np.int16)
            img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_arr)
            img.save(cls_dir / f"mock_{cls_name.lower().replace(' ', '_')}_{i:03d}.jpg")

    print(f"[OK] Created synthetic test dataset in {config.RAW_DATA_DIR}.")
    print("Now splitting mock dataset into stratified splits...")
    dataset.create_stratified_splits(
        raw_data_dir=config.RAW_DATA_DIR,
        output_dir=config.SPLIT_DATA_DIR,
        max_samples_per_class=config.SAMPLES_PER_CLASS,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Preparation & Splitting Utility")
    parser.add_argument("--setup", action="store_true", help="Create folder structure in data/raw/")
    parser.add_argument("--split", action="store_true", help="Verify and stratify raw dataset")
    parser.add_argument("--mock", action="store_true", help="Generate synthetic images to test the pipeline")
    parser.add_argument("--mock_count", type=int, default=15, help="Number of synthetic images per class for --mock (default: 15)")
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=config.SAMPLES_PER_CLASS,
        help="Max images per class to balance dataset (default: 470, set 0 for all)",
    )

    args = parser.parse_args()
    cap = args.samples_per_class if args.samples_per_class and args.samples_per_class > 0 else None

    if args.mock:
        generate_mock_dataset(num_samples_per_class=args.mock_count)
    elif args.split:
        verify_and_split(samples_per_class=cap)
    elif args.setup:
        setup_class_directories()
    else:
        # Default behavior: setup directories if missing, or split if present
        setup_class_directories()
