# Automated Multi-Class Coconut Tree Disease Detection, Diagnosis & Symptom Localization Using Deep Learning and Explainable AI (XAI)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%20CUDA%2012.1-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.97%25%20(MobileNetV2)-success.svg)](#-experimental-results--benchmark-leaderboard)
[![Edge Ready](https://img.shields.io/badge/Edge%20Inference-203.3%20FPS%20(8.5MB)-brightgreen.svg)](#hardware-efficiency--deployment-feasibility)

**B.Tech Final Year Project (BTP)**  
*Department of Computer Science and Engineering*  
**Indian Institute of Information Technology Kottayam (IIIT Kottayam)**

---

## 👥 Project Team Members

| Name | Roll Number | Email / GitHub |
| :--- | :---: | :---: |
| **Ansh Singh** | `2023BCS0120` | [2023bcs0120@iiitkottayam.ac.in](mailto:2023bcs0120@iiitkottayam.ac.in) |
| **Lakshmi Harshitha Cheedella** | `2023BCS0069` | [2023bcs0069@iiitkottayam.ac.in](mailto:2023bcs0069@iiitkottayam.ac.in) |
| **Soumalya Sarkar** | `2023BCS0172` | [2023bcs0172@iiitkottayam.ac.in](mailto:2023bcs0172@iiitkottayam.ac.in) |
| **Chennupati Bhargav** | `2023BCS0219` | [2023bcs0219@iiitkottayam.ac.in](mailto:2023bcs0219@iiitkottayam.ac.in) |

---

## 📌 Executive Summary & Abstract

Coconut palm (*Cocos nucifera*) is an indispensable plantation crop vital to the agrarian economies of tropical and coastal regions worldwide. However, plantations suffer devastating economic losses from infectious fungal, bacterial, and physiological pathologies. In particular, **Bud Rot** (*Phytophthora palmivora*) causes complete vascular collapse and irreversible tree mortality within weeks if unmanaged.

Conventional disease diagnosis relies on manual visual inspection by experienced arborists, a method constrained by:
1. Hazardous canopy heights (15–30 meters).
2. Human fatigue and visual subjectivity.
3. Fine-grained visual overlap (e.g., distinguishing **Leaf Rot** from **Gray Leaf Spot** necrotic lesions).
4. Severe natural class imbalance across field conditions.

This project implements an end-to-end, high-performance deep learning framework for **automated multi-class coconut tree disease classification and symptom localization**. Leveraging pre-trained convolutional neural backbones (MobileNetV2, ResNet-50, EfficientNet-B0), imbalance-mitigating sampling techniques (WeightedRandomSampler, Cost-Sensitive Weighted Cross-Entropy, Focal Loss), and a **Two-Stage Fine-Tuning Pipeline**, the optimized system achieves **98.97% overall test accuracy**, a **0.9916 Macro F1-Score**, and **100% recall on fatal Bud Rot** on unseen field imagery at **203.3 FPS**. Furthermore, **Grad-CAM (Gradient-Weighted Class Activation Mapping)** is integrated to deliver visual interpretability, allowing agronomists to audit the exact symptom regions that drive the model's predictions.

---

## 🔬 Key Features & Technical Highlights

- **Multi-Class Disease Coverage**: Classifies all 5 major coconut palm conditions:
  - `Bud Root Dropping` (Premature fruitlet and button shedding)
  - `Bud Rot` (Fatal meristematic heart rot caused by *Phytophthora*)
  - `Gray Leaf Spot` (*Pestalotiopsis palmarum* foliar blight)
  - `Leaf Rot` (Foliar collapse and flaccid leaflet decay)
  - `Stem Bleeding` (*Ceratocystis paradoxa* dark trunk exudate)
- **Class Imbalance Mitigation**:
  - Implements `WeightedRandomSampler` to guarantee equal class representation ($20\%$ per category) in every mini-batch.
  - Supports **Cost-Sensitive Weighted Cross-Entropy** with inverse-frequency penalties and **Focal Loss** ($\gamma = 2.0$) to prioritize hard minority samples.
  - Stratified partitioning ensuring consistent evaluation without minority depletion.
- **Two-Stage Fine-Tuning Pipeline**:
  - **Stage 1 (Warmup)**: Classification head warmup with frozen convolutional backbones to preserve ImageNet representations.
  - **Stage 2 (Full Fine-Tuning)**: End-to-end adaptation with differential learning rates (Backbone $\text{LR} = 10^{-5}$, Head $\text{LR} = 10^{-4}$) and Cosine Annealing decay.
- **Soft-Voting Ensemble Strategy**:
  - Probability-level blending combining the deep representational capacity of **ResNet-50** with the compact generalization of **MobileNetV2**.
- **Explainable AI (XAI)**:
  - Generates high-resolution **Grad-CAM** heatmaps overlaid on input imagery, highlighting pathological lesion boundaries for transparent clinical validation.
- **Edge Deployment Ready**:
  - MobileNetV2 delivers **191.9 FPS** with an ultra-compact **8.51 MB** footprint, enabling real-time deployment on drones, Raspberry Pi, and Android edge devices.

---

## 📊 Dataset Distribution & Architecture

The benchmark dataset comprises **5,798 field images** across all five disease classes:

| Class Label | Description / Pathology | Total Images | Natural % | Imbalance Status |
| :--- | :--- | :---: | :---: | :---: |
| **Gray Leaf Spot** | Elliptical lesions with grayish centers and dark borders | 2,135 | 36.8% | Majority |
| **Leaf Rot** | Blackening, rotting, and distal shriveling of leaflets | 1,673 | 28.9% | Majority |
| **Stem Bleeding** | Dark reddish-brown exudation oozing from trunk cracks | 1,006 | 17.3% | Moderate |
| **Bud Root Dropping** | Premature button and immature nut detachment | 514 | 8.9% | Minority |
| **Bud Rot** | Heart rot of the apical bud, leading to crown collapse | 470 | 8.1% | **Critical Minority** |
| **Total** | | **5,798** | **100.0%** | **4.5 : 1 Ratio** |

---

## 📈 Experimental Results & Benchmark Leaderboard

### 🏆 1. State-of-the-Art Test Benchmark (Full Dataset, $N = 874$ Test Images)
Following Two-Stage Fine-Tuning (classification head warmup + differential learning rates), $256 \times 256$ multi-scale augmentations with `RandomErasing`, and Label Smoothing ($0.1$), the optimized **MobileNetV2** achieved a new milestone on the test split:

| Model Architecture | Test Accuracy (%) | Macro F1 | Weighted F1 | Macro Recall | Macro Precision | Latency | Throughput | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **MobileNetV2 (Two-Stage + Full Data)** | **98.97%** | **0.9916** | **0.9897** | **99.37%** | **98.96%** | **4.92 ms** | **203.3 FPS** | **8.51 MB** |
| 🥈 **ResNet-50 (Baseline)** | 94.65% | 0.9459 | 0.9465 | 0.9465 | 0.9480 | 8.69 ms | 115.0 FPS | 89.72 MB |
| 🥉 **Soft-Voting Ensemble** | 93.92% | 0.9211 | 0.9407 | 96.39% | 89.67% | 11.08 ms | 90.3 FPS | 98.23 MB |

#### Per-Class Diagnostic Breakdown (MobileNetV2 — 874 Test Images)
| Disease Class | Support (Images) | Precision | Recall | F1-Score | Clinical & Pathology Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Bud Root Dropping** | 78 | **1.0000** | **1.0000** | **1.0000** | **100% Perfect Precision & Recall** |
| **Bud Rot** | 71 | **0.9726** | **1.0000** | **0.9861** | **Zero Missed Infections (71/71 caught)** |
| **Stem Bleeding** | 152 | **1.0000** | **1.0000** | **1.0000** | **100% Perfect Precision & Recall** |
| **Gray Leaf Spot** | 321 | **0.9875** | **0.9844** | **0.9860** | Highly robust on complex foliar lesions |
| **Leaf Rot** | 252 | **0.9880** | **0.9841** | **0.9861** | High discrimination against necrosis overlap |

> **Critical Agronomic Finding**: MobileNetV2 achieved a **100.0% Recall on the fatal Bud Rot pathology** (71 out of 71 infected test palms correctly detected with zero false negatives). Detecting 100% of fatal heart rot cases while sustaining **98.97% overall accuracy** at **203 FPS** establishes an ultra-reliable, edge-deployable solution for field inspection.

---

### 2. Large-Scale Out-of-Sample Field Evaluation ($N = 3,781$ Unseen Images)
To test true out-of-distribution robustness, models trained on balanced subsets were evaluated on the **3,781 remaining uncurated field images**:

| Model Architecture | Accuracy (%) | Macro F1 | Weighted F1 | Macro Recall | Macro Precision | Latency | Throughput | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Soft-Voting Ensemble** | **93.92%** | **0.9211** | **0.9407** | **96.39%** | 89.67% | 11.08 ms | 90.3 FPS | 98.2 MB |
| 🥈 **MobileNetV2** | **93.10%** | **0.9177** | **0.9321** | 95.07% | **90.12%** | **5.21 ms** | **191.9 FPS** | **8.5 MB** |
| 🥉 **ResNet-50** | 91.93% | 0.8844 | 0.9236 | **95.56%** | 85.03% | 8.70 ms | 115.0 FPS | 89.7 MB |

### 3. Per-Class Performance Breakdown (Soft-Voting Ensemble on Out-of-Sample Data)
| Disease Class | Support (Images) | Precision | Recall | F1-Score | Clinical Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Stem Bleeding** | 589 | **1.0000** | **1.0000** | **1.0000** | **100% Perfect Detection** |
| **Bud Root Dropping** | 140 | 0.9517 | **0.9857** | **0.9684** | Exceptional sensitivity |
| **Bud Rot** | 125 | 0.6250 | **1.0000** | 0.7692 | **Zero Missed Cases (125/125 caught)** |
| **Gray Leaf Spot** | 1,692 | **0.9284** | 0.9504 | 0.9393 | Highly robust foliar detection |
| **Leaf Rot** | 1,235 | **0.9785** | 0.8834 | 0.9285 | High precision on necrotic fronds |

> **Key Agricultural Finding**: The Soft-Voting Ensemble achieved a **100% Recall on Bud Rot** (125 out of 125 field cases detected). In plant pathology, false negatives on Bud Rot cause total tree mortality; thus, prioritizing perfect recall while maintaining 93.92% overall accuracy provides an optimal clinical safety net.

---

## 🛠️ Project Directory Structure

```text
c:/Github/BTP/
│
├── config.py                 # Centralized configuration, hyperparams, class definitions, and paths
├── dataset.py                # PyTorch Dataset, stratified split generation, and WeightedRandomSampler
├── models.py                 # Neural backbones factory (ResNet50, MobileNetV2, EfficientNet, DenseNet)
├── loss.py                   # Cost-sensitive loss implementations (Weighted Cross-Entropy, Focal Loss)
├── metrics.py                # Imbalance-aware evaluation metrics (Macro/Weighted F1, Per-class Recall)
├── train.py                  # End-to-end training pipeline with Two-Stage warmup and early stopping
├── evaluate.py               # Comprehensive evaluation runner, benchmark leaderboard, and Ensemble
├── explainability.py         # Grad-CAM localization engine generating overlaid visual saliency maps
├── download_dataset.py       # Automated Kaggle dataset downloader and synchronizer
├── prepare_data.py           # Dataset integrity checks and stratified partitioning utility
├── requirements.txt          # Python dependency specifications
│
├── data/
│   ├── raw/                  # Full raw dataset organized by class subfolders
│   └── splits/               # Generated stratified CSV splits (train.csv, val.csv, test.csv)
│
└── outputs/
    ├── checkpoints/          # Saved model weights (*.pth) for all architectures
    ├── metrics/              # JSON metrics logs and comparative CSV leaderboards
    └── plots/                # Training curves, confusion matrices, and Grad-CAM heatmaps
```

---

## 🚀 Installation & Quickstart

### 1. Prerequisites & Environment Setup
Clone the repository and create an isolated Python virtual environment:

```powershell
# Clone repository
git clone https://github.com/YourUsername/Coconut-Tree-Disease-Detection.git
cd Coconut-Tree-Disease-Detection

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies with PyTorch CUDA 12.1 acceleration
pip install -r requirements.txt
```

*(Ensure PyTorch detects your GPU)*:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```

### 2. Dataset Setup
Download and synchronize the Kaggle coconut disease dataset:
```powershell
python download_dataset.py
```

---

## 💻 Usage & CLI Guide

### 🏋️ Training Models

#### Train MobileNetV2 with Two-Stage Fine-Tuning & Full Dataset Sampling:
```powershell
python train.py --model mobilenet_v2 --resplit --epochs 25
```

#### Train ResNet-50 with Focal Loss:
```powershell
python train.py --model resnet50 --loss focal --epochs 25
```

#### Key Training Flags:
- `--model`: Backbone architecture (`mobilenet_v2`, `resnet50`, `efficientnet_b0`, `densenet121`).
- `--loss`: Loss function (`weighted_ce`, `focal`, `standard_ce`).
- `--epochs`: Total training epochs (default: `25`).
- `--warmup_epochs`: Number of initial epochs with frozen backbone (default: `3`).
- `--label_smoothing`: Label smoothing parameter (default: `0.1`).
- `--resplit`: Regenerates stratified train/val/test splits.
- `--no_sampler`: Disables WeightedRandomSampler (uses standard shuffle).

---

### 📊 Model Evaluation & Benchmarking

#### 1. Evaluate on the Standard Test Split:
```powershell
# Evaluate single model
python evaluate.py --model mobilenet_v2

# Evaluate all trained models and generate comparison table
python evaluate.py --all
```

#### 2. Evaluate on All Remaining Unseen Field Images (3,781 images):
```powershell
python evaluate.py --remaining
```

#### 3. Run Soft-Voting Ensemble:
```powershell
python evaluate.py --ensemble --remaining
```

All evaluation runs automatically generate:
- High-resolution confusion matrix heatmaps in `outputs/plots/cm_*.png`.
- Detailed per-class precision, recall, and F1 reports in `outputs/metrics/eval_*.json`.
- Comparative ranking table in `outputs/metrics/model_comparison_*.csv`.

---

### 🔍 Explainable AI (Grad-CAM Saliency Maps)

To visualize the exact lesion features influencing the CNN's decision:

```powershell
# Generate Grad-CAM for a specific field image
python explainability.py --model mobilenet_v2 --image path/to/leaf_image.jpg

# Generate Grad-CAM heatmaps across sample images for all classes
python explainability.py --all_classes
```

Grad-CAM highlights:
- Necrotic lesions and yellow chlorotic halos for **Gray Leaf Spot** and **Leaf Rot**.
- Dark trunk bleeding cracks for **Stem Bleeding**.
- Apical bud vascular decay for **Bud Rot**.

---

## ⚡ Hardware Efficiency & Deployment Feasibility

| Deployment Target | Recommended Architecture | Latency | Frame Rate | Footprint | Power Profile |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Drone / UAV Inspection** | MobileNetV2 | **4.92 ms** | **203.3 FPS** | **8.5 MB** | Ultra-Low Power |
| **Android / iOS Mobile App** | MobileNetV2 (ONNX/TFLite) | **~10 ms** | **~100 FPS** | **8.5 MB** | Battery Efficient |
| **Cloud Server Batch Diagnosis** | Soft-Voting Ensemble | **11.08 ms** | **90.3 FPS** | **98.2 MB** | High Redundancy / Safe |

---

## 📚 References & Academic Citations

1. **Dataset Source**: Thite, S., et al. (2023). *Multi-Class Coconut Tree Disease Dataset*. Kaggle.
2. **Focal Loss**: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal loss for dense object detection*. IEEE International Conference on Computer Vision (ICCV), 2980-2988.
3. **Grad-CAM**: Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization*. IEEE ICCV, 618-626.
4. **MobileNetV2**: Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted residuals and linear bottlenecks*. IEEE CVPR, 4510-4520.
5. **ResNet**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. IEEE CVPR, 770-778.

---

## 🎓 Acknowledgments

We express our sincere gratitude to our project supervisor, faculty members, and the **Department of Computer Science and Engineering at IIIT Kottayam** for their guidance, computational infrastructure, and encouragement throughout our final year B.Tech Project.
