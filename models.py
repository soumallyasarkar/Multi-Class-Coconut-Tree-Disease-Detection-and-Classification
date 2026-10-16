"""
Candidate Deep Learning Architectures for Model Benchmarking
Multi-Class Coconut Tree Disease Detection & Classification

Supports architectures evaluated in the literature review & architecture slide:
- MobileNetV2 (Lightweight Edge model)
- ResNet50 (Standard robust baseline)
- EfficientNet-B0 (Compound-scaled efficient network)
- DenseNet121 (Dense connectivity for fine-grained leaf pathology)
- VGG16 (Traditional comparative baseline)
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torchvision import models

import config


def build_model(
    model_name: str = "resnet50",
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: Optional[float] = None,
) -> nn.Module:
    """
    Factory function returning the specified pre-trained CNN architecture
    with its classification head adapted for the 5 coconut disease classes.
    """
    model_name = model_name.lower().strip()

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        p = dropout if dropout is not None else 0.3
        model.fc = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        p = dropout if dropout is not None else 0.2
        model.classifier = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(in_features, num_classes)
        )

    elif model_name in ["efficientnet_b0", "efficientnet"]:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        p = dropout if dropout is not None else 0.3
        model.classifier = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        p = dropout if dropout is not None else 0.3
        model.classifier = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        in_features = model.classifier[6].in_features
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        p = dropout if dropout is not None else 0.4
        model.classifier[6] = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(in_features, num_classes)
        )

    else:
        supported = ["resnet50", "mobilenet_v2", "efficientnet_b0", "densenet121", "vgg16"]
        raise ValueError(f"Unknown model '{model_name}'. Supported models: {supported}")

    return model


def get_model_summary(model: nn.Module) -> Dict[str, Union[int, float]]:
    """
    Returns parameter counts and estimated model memory footprint in MB.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Approximate model file size in Megabytes (float32 = 4 bytes per param)
    size_mb = total_params * 4.0 / (1024 * 1024)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb": round(size_mb, 2),
    }


if __name__ == "__main__":
    print("Testing Model Factory & Parameter Profiles:")
    for name in ["mobilenet_v2", "resnet50", "efficientnet_b0"]:
        net = build_model(name, pretrained=False)
        info = get_model_summary(net)
        print(f"  [{name:15s}] Params: {info['total_params']:,} | Size: {info['size_mb']} MB")
