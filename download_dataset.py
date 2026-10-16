 """
Download Coconut Tree Disease Dataset from Kaggle using kagglehub
and organize into data/raw/<class_name>/ structure.
"""

import shutil
from pathlib import Path
import config


def download_and_setup_dataset():
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Please install it with: pip install kagglehub")
        return

    print("=" * 65)
    print("DOWNLOADING DATASET FROM KAGGLE: nagabushan/coconut-tree-disease-dataset")
    print("=" * 65)

    downloaded_path = kagglehub.dataset_download("nagabushan/coconut-tree-disease-dataset")
    print(f"\n[OK] Kaggle download completed at: {downloaded_path}")

    source_dir = Path(downloaded_path)
    target_raw_dir = config.RAW_DATA_DIR
    target_raw_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing mock images if present
    print("\nScanning destination folder for mock synthetic images...")
    mock_files = list(target_raw_dir.glob("**/mock_*.jpg"))
    if mock_files:
        print(f"Removing {len(mock_files)} placeholder mock images from data/raw/...")
        for mf in mock_files:
            try:
                mf.unlink()
            except Exception as e:
                print(f"  Warning: could not remove {mf}: {e}")

    # Inspect downloaded contents
    print("\nInspecting downloaded files and mapping classes...")
    copied_counts = {cls_name: 0 for cls_name in config.CLASSES}

    # Find all image files in the downloaded source
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_downloaded_images = [
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]
    print(f"Total valid images discovered in Kaggle download: {len(all_downloaded_images)}")

    for img_path in all_downloaded_images:
        parent_folder_name = img_path.parent.name.strip().lower().replace("_", " ")

        # Match to one of the 5 project classes
        matched_class = None
        for cls_name in config.CLASSES:
            norm_cls = cls_name.lower().replace("_", " ")
            if norm_cls == parent_folder_name or norm_cls in parent_folder_name:
                matched_class = cls_name
                break

        if matched_class is not None:
            dest_dir = target_raw_dir / matched_class
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / img_path.name

            # Copy file if not already present
            if not dest_file.exists():
                shutil.copy2(img_path, dest_file)
            copied_counts[matched_class] += 1
        else:
            print(f"  [Unmapped Image] {img_path.relative_to(source_dir)}")

    print("\n" + "=" * 65)
    print("DATASET SYNC TO data/raw/ COMPLETE")
    print("=" * 65)
    total_synced = sum(copied_counts.values())
    for cls_name, cnt in copied_counts.items():
        expected = config.BENCHMARK_CLASS_COUNTS.get(cls_name, 0)
        print(f"  - {cls_name:20s}: {cnt:5d} images (Benchmark: {expected})")
    print(f"\nTotal synced images: {total_synced}")

    if total_synced > 0:
        print("\nNext step: Run 'python prepare_data.py --split' to partition into 70% train / 15% val / 15% test.")


if __name__ == "__main__":
    download_and_setup_dataset()
