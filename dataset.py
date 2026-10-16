"""
Dataset Management, Stratified Splitting, and Imbalance-Aware Sampling
Multi-Class Coconut Tree Disease Detection & Classification
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import config


# ====================================================================
# 1. Image Transforms (Data Augmentation & Normalization)
# ====================================================================

def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Returns image transformation pipelines.
    - 'train': Includes geometric & photometric augmentations, adaptive crops,
               and RandomErasing to force multi-lesion feature learning.
    - 'val' / 'test': Deterministic resizing and ImageNet standard normalization.
    """
    # ImageNet standard statistics used by all pre-trained vision backbones
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    crop_h, crop_w = config.IMAGE_SIZE
    base_resize = int(crop_h * 1.125)

    if split == "train":
        return transforms.Compose([
            transforms.Resize((base_resize, base_resize)),
            transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value="random"),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((base_resize, base_resize)),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ])


# ====================================================================
# 2. PyTorch Dataset Implementation
# ====================================================================

class CoconutDataset(Dataset):
    """
    Loads coconut disease images and labels.
    Accepts either a pandas DataFrame with columns ['image_path', 'label']
    or a directory path structured as: root/class_name/image.jpg
    """

    def __init__(
        self,
        data_source: Union[pd.DataFrame, str, Path],
        transform: Optional[transforms.Compose] = None,
    ):
        self.transform = transform

        if isinstance(data_source, pd.DataFrame):
            self.df = data_source.reset_index(drop=True)
        elif isinstance(data_source, (str, Path)):
            self.df = self._scan_directory(Path(data_source))
        else:
            raise TypeError("data_source must be a pandas DataFrame or directory Path.")

        # Class names to numeric mapping
        self.classes = config.CLASSES
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Validate that labels are integer indices
        if "label" not in self.df.columns and "class_name" in self.df.columns:
            self.df["label"] = self.df["class_name"].map(self.class_to_idx)

        # Cache targets as numpy array for fast sampling weight calculations
        self.targets = self.df["label"].values.astype(int)

    def _scan_directory(self, root_dir: Path) -> pd.DataFrame:
        records = []
        for class_name in config.CLASSES:
            # Handle possible naming variations (lowercase, underscores)
            folder_candidates = [
                root_dir / class_name,
                root_dir / class_name.lower(),
                root_dir / class_name.lower().replace(" ", "_"),
                root_dir / class_name.replace(" ", "_"),
            ]
            class_folder = next((p for p in folder_candidates if p.exists() and p.is_dir()), None)

            if class_folder is not None:
                seen_paths = set()
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                    for img_path in class_folder.glob(ext):
                        norm_path = str(img_path.resolve())
                        if norm_path.lower() not in seen_paths:
                            seen_paths.add(norm_path.lower())
                            records.append({
                                "image_path": norm_path,
                                "class_name": class_name,
                                "label": config.CLASSES.index(class_name),
                            })

        if not records:
            raise FileNotFoundError(f"No valid images found in directory structure: {root_dir}")
        return pd.DataFrame(records)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = int(row["label"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupt images: create a blank image to avoid crashing training
            print(f"[Warning] Corrupted image skipped/blanked: {img_path} ({e})")
            image = Image.new("RGB", config.IMAGE_SIZE, (0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_class_counts(self) -> Dict[str, int]:
        """Returns the sample count per class present in this dataset."""
        counts = {}
        for idx, cls_name in enumerate(self.classes):
            counts[cls_name] = int((self.targets == idx).sum())
        return counts


# ====================================================================
# 3. Stratified Dataset Splitting (Prevents Minority Depletion)
# ====================================================================

def create_stratified_splits(
    raw_data_dir: Path = config.RAW_DATA_DIR,
    output_dir: Path = config.SPLIT_DATA_DIR,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    random_seed: int = config.RANDOM_SEED,
    max_samples_per_class: Optional[int] = config.SAMPLES_PER_CLASS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Discovers all images in raw_data_dir and partitions them into Train, Val, and Test
    using Stratified Sampling.
    
    If max_samples_per_class is set (e.g., 470), each class is capped/undersampled
    to that number of images, creating a balanced dataset across all categories.
    
    Generates CSV metadata files in output_dir (saving disk space by not copying images).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    full_dataset = CoconutDataset(raw_data_dir)
    df = full_dataset.df.copy()

    np.random.seed(random_seed)

    train_rows, val_rows, test_rows = [], [], []

    print(f"\nPartitioning data (max_samples_per_class={max_samples_per_class}):")

    # Stratify by each class individually
    for cls_idx, cls_name in enumerate(config.CLASSES):
        class_df = df[df["label"] == cls_idx].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        raw_count = len(class_df)

        if max_samples_per_class is not None and raw_count > max_samples_per_class:
            class_df = class_df.iloc[:max_samples_per_class].reset_index(drop=True)
            print(f"  [Balanced Cap] '{cls_name:20s}': capped from {raw_count:4d} down to {len(class_df):4d} images")
        else:
            print(f"  [Retained]     '{cls_name:20s}': using {raw_count:4d} images" +
                  (f" (target cap: {max_samples_per_class})" if max_samples_per_class and raw_count < max_samples_per_class else ""))

        n = len(class_df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_rows.append(class_df.iloc[:n_train])
        val_rows.append(class_df.iloc[n_train:n_train + n_val])
        test_rows.append(class_df.iloc[n_train + n_val:])

    train_df = pd.concat(train_rows).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    val_df = pd.concat(val_rows).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    test_df = pd.concat(test_rows).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Save split metadata to disk
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    total_used = len(train_df) + len(val_df) + len(test_df)
    print(f"\nStratified Splits Saved to {output_dir}:")
    print(f"  Total images used: {total_used} (out of {len(df)} available raw images)")
    print(f"  Train: {len(train_df)} images ({len(train_df)/total_used*100:.1f}%)")
    print(f"  Val:   {len(val_df)} images ({len(val_df)/total_used*100:.1f}%)")
    print(f"  Test:  {len(test_df)} images ({len(test_df)/total_used*100:.1f}%)")

    return train_df, val_df, test_df


# ====================================================================
# 4. Balanced Sampler for Handling Imbalance
# ====================================================================

def create_balanced_sampler(dataset: CoconutDataset) -> WeightedRandomSampler:
    """
    WeightedRandomSampler gives each sample a probability of being selected inversely
    proportional to its class frequency.
    
    Result: Every mini-batch contains an approximately equal number of images from
    each of the 5 classes, completely neutralizing gradient bias toward majority classes.
    """
    class_counts = np.bincount(dataset.targets, minlength=len(config.CLASSES))
    class_weights = 1.0 / np.maximum(class_counts, 1)

    # Assign weight to each individual sample
    sample_weights = [class_weights[target] for target in dataset.targets]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,  # minority samples are re-sampled with replacement
    )
    return sampler


# ====================================================================
# 5. Master DataLoader Factory
# ====================================================================

def get_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    use_balanced_sampler: bool = config.USE_BALANCED_SAMPLER,
    num_workers: int = config.NUM_WORKERS,
    force_resplit: bool = False,
    max_samples_per_class: Optional[int] = config.SAMPLES_PER_CLASS,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Returns train_loader, val_loader, test_loader, and train_class_counts.
    Looks for split CSV files in config.SPLIT_DATA_DIR first;
    if absent or if force_resplit=True, generates them from config.RAW_DATA_DIR.
    """
    train_csv = config.SPLIT_DATA_DIR / "train.csv"
    val_csv = config.SPLIT_DATA_DIR / "val.csv"
    test_csv = config.SPLIT_DATA_DIR / "test.csv"

    if force_resplit or not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        if config.RAW_DATA_DIR.exists() and any(config.RAW_DATA_DIR.iterdir()):
            print("Generating stratified splits from raw data directory...")
            train_df, val_df, test_df = create_stratified_splits(
                raw_data_dir=config.RAW_DATA_DIR,
                output_dir=config.SPLIT_DATA_DIR,
                max_samples_per_class=max_samples_per_class,
            )
        else:
            raise FileNotFoundError(
                f"Split files not found in {config.SPLIT_DATA_DIR} and raw data not found in {config.RAW_DATA_DIR}.\n"
                f"Please place your dataset images in '{config.RAW_DATA_DIR}/<class_name>/' or run the mock generator."
            )
    else:
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)

    train_dataset = CoconutDataset(train_df, transform=get_transforms("train"))
    val_dataset = CoconutDataset(val_df, transform=get_transforms("val"))
    test_dataset = CoconutDataset(test_df, transform=get_transforms("test"))

    # Sampler or standard shuffle
    if use_balanced_sampler:
        sampler = create_balanced_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_class_counts = train_dataset.get_class_counts()
    return train_loader, val_loader, test_loader, train_class_counts


# ====================================================================
# 6. Remaining Out-of-Sample Images (Images outside 470-sample splits)
# ====================================================================

def get_remaining_dataset(
    raw_data_dir: Path = config.RAW_DATA_DIR,
    split_data_dir: Path = config.SPLIT_DATA_DIR,
    save_csv: bool = True,
) -> CoconutDataset:
    """
    Returns a CoconutDataset containing all raw images that were NOT included
    in train.csv, val.csv, or test.csv (unseen out-of-sample images).
    """
    remaining_csv = split_data_dir / "remaining.csv"
    if remaining_csv.exists() and not save_csv:
        df = pd.read_csv(remaining_csv)
    else:
        full_dataset = CoconutDataset(raw_data_dir)
        full_df = full_dataset.df

        used_paths = set()
        for split_name in ["train.csv", "val.csv", "test.csv"]:
            p = split_data_dir / split_name
            if p.exists():
                sdf = pd.read_csv(p)
                used_paths.update(sdf["image_path"].str.lower().tolist())

        df = full_df[~full_df["image_path"].str.lower().isin(used_paths)].copy().reset_index(drop=True)
        if save_csv:
            df.to_csv(remaining_csv, index=False)
            print(f"Saved {len(df)} remaining out-of-sample images to {remaining_csv}")

    return CoconutDataset(df, transform=get_transforms("test"))


def get_remaining_loader(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    save_csv: bool = True,
) -> DataLoader:
    """
    Returns a DataLoader for the remaining out-of-sample images.
    """
    rem_dataset = get_remaining_dataset(save_csv=save_csv)
    return DataLoader(
        rem_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

