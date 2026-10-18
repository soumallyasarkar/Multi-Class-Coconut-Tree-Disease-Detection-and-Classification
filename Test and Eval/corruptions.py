"""
Image Corruption and Damage Suite for Out-of-Distribution Robustness
Multi-Class Coconut Tree Disease Detection & Classification

Implements the 4 corruption families across 3 severities matching standard benchmarks:
1. Gaussian Blur (sigma: 1.0, 2.0, 4.0)
2. Darkness (brightness factor: 0.8x, 0.5x, 0.2x)
3. Brightness (brightness factor: 1.2x, 1.8x, 2.2x)
4. JPEG Compression (quality factor: 50, 20, 5)
"""

import io
from typing import Callable, Dict, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# Corruption Definitions and Parameters
CORRUPTIONS: Dict[str, Dict[int, float]] = {
    "blur": {1: 1.0, 2: 2.0, 3: 4.0},        # Gaussian blur radius (sigma)
    "dark": {1: 0.8, 2: 0.5, 3: 0.2},        # Brightness scale factor (< 1.0)
    "bright": {1: 1.2, 2: 1.8, 3: 2.2},      # Brightness scale factor (> 1.0)
    "jpeg": {1: 50, 2: 20, 3: 5},            # JPEG compression quality (1-100)
}

ALL_CONDITIONS = ["clean"] + [
    f"{family}_{sev}" for family in ["blur", "dark", "bright", "jpeg"] for sev in [1, 2, 3]
]


def apply_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Applies Gaussian Blur with standard deviation sigma."""
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_brightness(img: Image.Image, factor: float) -> Image.Image:
    """Adjusts image brightness using PIL ImageEnhance."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Simulates JPEG compression artifacts by encoding and decoding in memory."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def corrupt_image(img: Image.Image, corruption_name: str, severity: int) -> Image.Image:
    """
    Applies the specified corruption at given severity (1, 2, or 3) to a PIL Image.
    
    Args:
        img: Input PIL Image (RGB)
        corruption_name: 'blur', 'dark', 'bright', or 'jpeg'
        severity: 1 (mild), 2 (moderate), or 3 (extreme)
    """
    if corruption_name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption: {corruption_name}. Available: {list(CORRUPTIONS.keys())}")
    if severity not in [1, 2, 3]:
        raise ValueError(f"Severity must be 1, 2, or 3, got {severity}")

    param = CORRUPTIONS[corruption_name][severity]

    if corruption_name == "blur":
        return apply_blur(img, sigma=param)
    elif corruption_name in ["dark", "bright"]:
        return apply_brightness(img, factor=param)
    elif corruption_name == "jpeg":
        return apply_jpeg(img, quality=int(param))
    else:
        return img


class CorruptedDataset(Dataset):
    """
    PyTorch Dataset wrapper that applies corruption dynamically in memory.
    Avoids writing 12 duplicate copies of the dataset to disk.
    """

    def __init__(self, base_dataset: Dataset, corruption_name: str = "clean", severity: int = 1, transform: Callable = None):
        """
        Args:
            base_dataset: Underlying PyTorch dataset (yielding PIL Image or Tensor, label)
            corruption_name: 'clean', 'blur', 'dark', 'bright', or 'jpeg'
            severity: 1, 2, or 3
            transform: Final torchvision transforms (ToTensor, Normalize) applied after corruption
        """
        self.base_dataset = base_dataset
        self.corruption_name = corruption_name
        self.severity = severity
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.base_dataset[idx]
        
        # Handle different base dataset return structures
        if isinstance(item, (tuple, list)):
            img, label = item[0], item[1]
        else:
            img = item
            label = 0

        # Ensure image is in PIL format for corruption functions
        if isinstance(img, torch.Tensor):
            # If already tensor (C, H, W), convert to PIL
            if img.ndimension() == 3 and img.shape[0] in [1, 3]:
                # Assume unnormalized or denormalized
                to_pil = T.ToPILImage()
                img = to_pil(img)
            else:
                raise TypeError(f"Unexpected tensor shape: {img.shape}")
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        # Apply corruption if not clean
        if self.corruption_name != "clean":
            img = corrupt_image(img, self.corruption_name, self.severity)

        # Apply torchvision transforms
        if self.transform is not None:
            img = self.transform(img)

        return img, label
