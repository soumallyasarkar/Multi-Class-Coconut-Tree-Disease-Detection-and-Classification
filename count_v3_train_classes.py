"""
Class Distribution Profiler: Roboflow V3 Training Split (12,108 images)
Multi-Class Coconut Tree Disease Detection & Classification
"""

from collections import Counter
from pathlib import Path
import pandas as pd
import config

V3_DIR = config.DATA_DIR / "Coconut_Tree_Disease_Dataset.v3i.folder"
TRAIN_DIR = V3_DIR / "train"
TRAIN_CSV = config.SPLIT_DATA_DIR / "train_v3.csv"

PREFIX_MAP = [
    ("BudRootDropping", "Bud Root Dropping"),
    ("BudRot", "Bud Rot"),
    ("GrayLeafSpot", "Gray Leaf Spot"),
    ("LeafRot", "Leaf Rot"),
    ("StemBleeding", "Stem Bleeding"),
]


def count_train_classes():
    counts = Counter()
    
    # 1. Check if cached CSV exists
    if TRAIN_CSV.exists():
        print(f"Reading cached training split from: {TRAIN_CSV}")
        df = pd.read_csv(TRAIN_CSV)
        for _, row in df.iterrows():
            counts[row["class_name"]] += 1
    # 2. Otherwise scan the train directory directly
    elif TRAIN_DIR.exists():
        print(f"Scanning directory: {TRAIN_DIR}")
        for img_path in TRAIN_DIR.rglob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                name = img_path.name
                for prefix, class_name in PREFIX_MAP:
                    if name.startswith(prefix):
                        counts[class_name] += 1
                        break
    else:
        raise FileNotFoundError(f"Neither {TRAIN_CSV} nor {TRAIN_DIR} could be found!")

    total_images = sum(counts.values())
    max_count = max(counts.values()) if counts else 1

    print("\n" + "=" * 76)
    print("      ROBOFLOW V3 DATASET: TRAINING SET CLASS DISTRIBUTION (12,108 IMAGES)")
    print("=" * 76)
    print(f"{'Idx':<4} | {'Disease Category':<22} | {'Count':<8} | {'Share (%)':<10} | {'Imbalance Ratio':<15}")
    print("-" * 76)

    for idx, class_name in enumerate(config.CLASSES):
        cnt = counts.get(class_name, 0)
        pct = (cnt / total_images * 100) if total_images > 0 else 0.0
        ratio = f"1 : {max_count / cnt:.2f}" if cnt > 0 else "N/A"
        print(f"{idx:<4} | {class_name:<22} | {cnt:<8d} | {pct:>8.2f}% | {ratio:<15}")

    print("-" * 76)
    print(f"{'TOTAL':<29} | {total_images:<8d} | {100.00:>8.2f}% |")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    count_train_classes()
