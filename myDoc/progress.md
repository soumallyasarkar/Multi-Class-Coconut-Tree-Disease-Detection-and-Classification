# Coconut Tree Disease Detection & Classification — Project Progress

**Project Title:** Multi-Class Coconut Tree Disease Detection & Classification  
**Academic Context:** B.Tech Project — IIIT Kottayam  
**Last Updated:** October 13, 2026  
**Status:** Phase 1–4 Codebase Implemented & Validated; Environment Configured  

---

## 1. Executive Summary

This project develops an end-to-end, imbalance-resilient Deep Learning system for classifying coconut tree diseases across 5 distinct diagnostic categories based on benchmark literature (*Thite et al., 2023*). 

The primary research challenges addressed are:
1. **Severe Class Imbalance (~4.5 : 1 ratio):** Mitigating gradient starvation and majority-class bias for critical minority diseases (*Bud Rot* at 8.1% and *Bud Root Dropping* at 8.9%).
2. **Multi-Architecture Benchmarking:** Comparing edge-lightweight vs. heavy backbones (*MobileNetV2*, *ResNet50*, *EfficientNet-B0*, *DenseNet121*, *VGG16*).
3. **Visual Explainability (Grad-CAM):** Answering *"can we localize the symptom?"* by generating class activation heatmaps over lesions and rot spots.

---

## 2. Dataset Profile & Disease Classes

The benchmark distribution comprises **5,798 images** across 5 categories:

| Class Name | Benchmark Count | Benchmark % | Category Profile | Typical Symptom Region |
| :--- | :---: | :---: | :--- | :--- |
| **Gray Leaf Spot** | 2,135 | 36.8% | Majority | Leaves (circular grey/brown necrotic spots) |
| **Leaf Rot** | 1,673 | 28.9% | Majority | Leaves (distal leaf blackened decay) |
| **Stem Bleeding** | 1,006 | 17.3% | Moderate | Trunk/Bark (reddish-brown liquid exudation) |
| **Bud Root Dropping** | 514 | 8.9% | **Minority** | Crown base & nut stalks |
| **Bud Rot** | 470 | 8.1% | **Minority (Critical)** | Central spindle leaf & growing bud crown |

---

## 3. The 4 Project Phases & Current Milestones

```mermaid
flowchart LR
    A["Phase 1: Data Pipeline & Environment"] --> B["Phase 2: Imbalance Mitigation & Models"]
    B --> C["Phase 3: Evaluation & Benchmarking"]
    C --> D["Phase 4: Explainability & Deployment"]
```

### Phase 1: Environment Setup & Data Pipeline Foundation
- [x] **Isolated Virtual Environment (`.venv`):** Configured local virtual environment inside `c:\Github\BTP\.venv` with all project dependencies installed.
- [x] **Directory Architecture:** Automated creation of `data/raw/`, `data/splits/`, `outputs/checkpoints/`, `outputs/metrics/`, and `outputs/plots/`.
- [x] **Stratified Splitting:** Created a deterministic 70% Train / 15% Validation / 15% Test stratified partition that strictly preserves minority class representation without data leakage.
- [x] **Synthetic Mock Dataset Generator:** Built `--mock` generator to test the full pipeline end-to-end before loading the full raw dataset.

### Phase 2: Imbalance Mitigation & Deep Learning Backbones
- [x] **Triple-Strategy Imbalance Mitigation:**
  1. **Sampling:** `WeightedRandomSampler` dynamically balances every mini-batch so minority examples are sampled with equal frequency.
  2. **Cost-Sensitive Loss:** Normalized inverse class-frequency weights penalize misclassifications on minority diseases.
  3. **Focal Loss:** Implementation of multi-class Focal Loss ($\gamma = 2.0$) down-weights easy majority examples and focuses gradients on hard minority cases.
- [x] **Candidate Model Architecture Suite:**
  - `MobileNetV2`: Ultra-lightweight edge model (~3.5M params, ~14 MB) for mobile deployment.
  - `ResNet50`: Standard residual network baseline (~25.6M params, ~98 MB).
  - `EfficientNet-B0`: Compound-scaled efficient architecture (~5.3M params, ~21 MB).
  - `DenseNet121`: Feature reuse architecture (~7.9M params, ~32 MB) optimal for fine-grained leaf pathology.
  - `VGG16`: Classical comparative baseline.
- [x] **Standardized Training Pipeline:** Checkpoints models strictly on **Macro F1-Score** (not raw accuracy) to prevent selecting models that over-fit to majority classes.

### Phase 3: Comprehensive Evaluation & Benchmarking Suite
- [x] **Evaluation Protocol:**
  - Macro-averaged Precision, Recall, and F1-Score.
  - Detailed per-class diagnostic reports (specifically tracking *Bud Rot* sensitivity).
  - Automated Confusion Matrix heatmaps saved to disk.
- [x] **Inference Profiling:** Real-time single-image latency (ms) and throughput (FPS) measurement.
- [x] **Model Comparison Tool:** Automated generation of side-by-side benchmark comparison tables across architectures.

### Phase 4: Symptom Explainability (Grad-CAM) & Inference
- [x] **Grad-CAM Engine:** Visual symptom localization by computing gradients of the predicted class with respect to the final convolutional feature maps.
- [x] **Multi-Backbone Target Hooking:** Automatically attaches forward and backward hooks to appropriate layers across ResNet, MobileNet, EfficientNet, DenseNet, and VGG.
- [x] **Single-Image & Batch Predictor:** CLI tool (`predict.py`) for individual disease diagnosis and batch processing with optional Grad-CAM visual output.

---

## 4. Codebase Architecture & File Analysis

| File | Purpose | Key Components & Functions |
| :--- | :--- | :--- |
| [`config.py`](file:///c:/Github/BTP/config.py) | Central Configuration | Class definitions, benchmark distributions, paths, training hyperparameters (`BATCH_SIZE=32`, `LR=1e-4`), imbalance settings. |
| [`dataset.py`](file:///c:/Github/BTP/dataset.py) | Data Ingestion & Sampler | `CoconutDataset`, `get_transforms()` (augmentations for training, standard normalization for val/test), `create_stratified_splits()`, `create_balanced_sampler()`, `get_dataloaders()`. |
| [`prepare_data.py`](file:///c:/Github/BTP/prepare_data.py) | Data Prep & Setup CLI | `--setup` (creates raw directories), `--split` (audits images and generates splits), `--mock` (generates synthetic images for testing). |
| [`models.py`](file:///c:/Github/BTP/models.py) | Neural Network Backbones | `build_model()` (factory for MobileNetV2, ResNet50, EfficientNet-B0, DenseNet121, VGG16 with custom heads), `get_model_summary()`. |
| [`loss.py`](file:///c:/Github/BTP/loss.py) | Imbalance Loss Functions | `compute_class_weights()` (`inverse_freq`, `effective_samples`, `sqrt_inverse`), `FocalLoss`, `get_loss_function()`. |
| [`metrics.py`](file:///c:/Github/BTP/metrics.py) | Imbalance-Aware Metrics | `compute_confusion_matrix()`, `compute_metrics()` (accuracy, macro/weighted precision, recall, F1, per-class metrics). |
| [`train.py`](file:///c:/Github/BTP/train.py) | Training & Validation Loop | `train_one_epoch()`, `evaluate_epoch()`, `train_model()`, Macro-F1 checkpoint saving, training curves plot generation (`curves_*.png`). |
| [`evaluate.py`](file:///c:/Github/BTP/evaluate.py) | Standalone Evaluation & Latency | `plot_confusion_matrix()`, `benchmark_latency()`, `evaluate_model()`, `benchmark_all_models()`. |
| [`explainability.py`](file:///c:/Github/BTP/explainability.py) | Grad-CAM Localization | `GradCAM` class (hooks, gradient pooling, ReLU, upsampling), `get_target_layer()`, `explain_image()` (heatmap overlay generation). |
| [`predict.py`](file:///c:/Github/BTP/predict.py) | Inference Pipeline | `predict_single_image()`, `predict_directory()`, CLI for top-$K$ predictions and Grad-CAM visualization. |
| [`requirements.txt`](file:///c:/Github/BTP/requirements.txt) | Environment Specification | PyTorch, TorchVision, NumPy, Pandas, Scikit-Learn, Pillow, Matplotlib, Seaborn, TQDM. |

---

## 5. Artifacts & Generated Files

- **`outputs/checkpoints/`**: Stores best model weights (`best_mobilenet_v2.pth` generated during pipeline test).
- **`outputs/metrics/`**: Stores training history logs in JSON format (`history_mobilenet_v2.json`).
- **`outputs/plots/`**: Stores training loss & Macro-F1 curves (`curves_mobilenet_v2.png`).
- **`data/splits/`**: Stores partition metadata (`train.csv`, `val.csv`, `test.csv`).

---

## 6. Next Steps & Recommended Actions

1. **Populate Real Dataset:**
   - Copy or download the full dataset images into `data/raw/<Class Name>/`.
   - Run `python prepare_data.py --split` to generate splits based on actual images.

2. **Execute Multi-Model Benchmarking:**
   - Train baseline models:
     ```bash
     python train.py --model mobilenet_v2 --epochs 20
     python train.py --model resnet50 --epochs 20
     python train.py --model efficientnet_b0 --epochs 20
     ```
   - Compare performance across architectures:
     ```bash
     python evaluate.py --benchmark_all
     ```

3. **Generate Explainability Visualizations:**
   - Run Grad-CAM on sample test images across all 5 disease categories to inspect disease localization accuracy.
   - Example:
     ```bash
     python predict.py --image path/to/sample.jpg --explain
     ```
