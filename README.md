# CocoGuard: Automated Multi-Class Coconut Tree Disease Detection, Out-of-Distribution Robustness & Explainable AI (XAI)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%20CUDA%2012.1-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Clean Zero-Shot Acc](https://img.shields.io/badge/Zero--Shot%20Acc-92.82%25%20(EfficientNet--B0)-success.svg)](#1-zero-shot-cross-plantation-benchmark-13836-unseen-field-images)
[![Edge Efficiency](https://img.shields.io/badge/Edge%20Inference-203.3%20FPS%20(8.5MB)-brightgreen.svg)](#hardware-efficiency--deployment-feasibility)
[![Selective Accuracy](https://img.shields.io/badge/Selective%20Acc%20@%2080%25-97.90%25-blueviolet.svg)](#3-probabilistic-calibration--uncertainty-rejection-selective-classification)
[![PDF Report](https://img.shields.io/badge/Publication%20Report-PDF%20Available-red.svg)](myDoc/CocoGuard_Research_Report.pdf)

**Final Year B.Tech Capstone Project (BTP)**  
*Department of Computer Science and Engineering*  
**Indian Institute of Information Technology Kottayam (IIIT Kottayam)**

---

## Project Team Members

| Name | Roll Number | Institutional Email |
| :--- | :---: | :---: |
| **Ansh Singh** | `2023BCS0120` | [2023bcs0120@iiitkottayam.ac.in](mailto:2023bcs0120@iiitkottayam.ac.in) |
| **Lakshmi Harshitha Cheedella** | `2023BCS0069` | [2023bcs0069@iiitkottayam.ac.in](mailto:2023bcs0069@iiitkottayam.ac.in) |
| **Soumallya Sarkar** | `2023BCS0172` | [2023bcs0172@iiitkottayam.ac.in](mailto:2023bcs0172@iiitkottayam.ac.in) |
| **Chennupati Bhargav** | `2023BCS0219` | [2023bcs0219@iiitkottayam.ac.in](mailto:2023bcs0219@iiitkottayam.ac.in) |

---

## 📚 Documentation & Research Report Hub

All detailed research findings, mathematical derivations, presentation materials, and technical guides are organized in the [`myDoc/`](myDoc/) directory:

| Document / Report | Description | Direct Link |
| :--- | :--- | :---: |
| 📄 **CocoGuard Comprehensive Research Report** | **Master empirical report:** 6 sections covering all 4 architectures, the 13,836-image field benchmark, 12 corruptions, sink class dynamics, selective classification ($97.9\%$ @ 80% coverage), and Grad-CAM stability. | [**Read Report (.md)**](myDoc/CocoGuard_Comprehensive_Research_Report.md) |
| 📑 **CocoGuard Publication PDF Report** | **Official academic PDF document (8 pages):** Formatted with running headers, light-blue shaded tables, callout cards, and all 9 authentic model-generated figures. | [**Download PDF (11.9 MB)**](myDoc/CocoGuard_Research_Report.pdf) |
| 📝 **Notes on Models & Grad-CAM** | Concise technical report structured alongside the PDF exhibits for fast reference and code mapping. | [**View Notes (.md)**](myDoc/research_report_models_and_gradcam.md) |
| 📊 **BTP Presentation Context** | Curated context document designed specifically for generating PowerPoint / Google Slides presentations for project reviews and seminars. | [**View Context (.md)**](myDoc/presentation_context_btp.md) |
| ⚖️ **Data Imbalance Mitigation Manifesto** | Mathematical formulation and PyTorch implementation of `WeightedRandomSampler` and cost-sensitive loss for overcoming the 4.5:1 class imbalance. | [**View Manifesto (.md)**](myDoc/data_imbalance_mitigation.md) |
| ⚙️ **Two-Stage Training Walkthrough** | Deep-dive into Stage 1 linear head warmup, Stage 2 differential learning rates, Cosine Annealing, and regularizers. | [**View Walkthrough (.md)**](myDoc/train_explanation.md) |
| 📈 **Project Milestones & Status** | Historical development tracking, phase completion records, and technical transition notes. | [**View Milestones (.md)**](myDoc/project_progress_and_current_status.md) \| [**Progress Notes (.md)**](myDoc/progress.md) |


---

## Executive Summary & Research Scope

Coconut palm (*Cocos nucifera*) plantations are vital to tropical agricultural economies, but suffer devastating yield losses from lethal pathologies—especially **Bud Rot** (*Phytophthora palmivora*), which destroys the apical bud and causes tree death within weeks if untreated.

While deep neural networks achieve high accuracy on standard test sets, deploying computer vision in agriculture exposes severe domain shifts: **variable sunlight, lens blur from tree swaying, cellular JPEG compression, and label confusion between foliar necrosis diseases**.

This repository implements **CocoGuard**, a comprehensive deep learning framework evaluating four distinct vision backbones:
1. [MobileNetV2](src/models.py): Inverted residuals and linear bottlenecks (ultra-efficient for edge/mobile devices).
2. [ResNet50](src/models.py): Standard deep residual baseline with identity shortcuts.
3. [EfficientNet-B0](src/models.py): Compound-scaled depth, width, and resolution architecture.
4. [DenseNet-121](src/models.py): Dense cross-layer connectivity with direct feature concatenation.

All models were trained strictly on a standardized 5,798-image dataset and then subjected to a **Zero-Shot Cross-Plantation Stress-Test across 13,836 unseen field images**, evaluating out-of-distribution robustness across 12 corruption conditions, post-hoc calibration, selective classification, and visual interpretability via **Grad-CAM**.

---

## Key Experimental Results & Discoveries

### 1. Zero-Shot Cross-Plantation Benchmark (13,836 Unseen Field Images)

Models were evaluated directly on 13,836 images from an independent field dataset without plantation-specific fine-tuning:

| Evaluation Metric | MobileNetV2 | ResNet50 | EfficientNet-B0 | DenseNet-121 |
| :--- | :---: | :---: | :---: | :---: |
| **Model Size (.pth)** | **8.51 MB** 🥇 | 89.72 MB | 15.31 MB 🥈 | 26.55 MB |
| **Parameter Count** | **2.23 M** 🥇 | 23.52 M | 4.01 M 🥈 | 6.96 M |
| **Clean Accuracy (13,836 Imgs)** | **88.35%** | 81.27% | **92.82%** 🥇 | **89.95%** 🥈 |
| **Clean Macro-F1 (95% Bootstrap CI)** | **0.8844** `[0.879, 0.890]` | **0.8089** `[0.801, 0.816]` | **0.9328** `[0.928, 0.937]` 🥇 | **0.9050** `[0.900, 0.910]` 🥈 |
| **Diagnostic Errors (of 13,836)** | 1,612 errors | 2,591 errors | **994 errors** 🥇 | 1,390 errors 🥈 |
| **Clean Expected Calibration Error (ECE)** | **0.1607** 🥇 | 0.2494 | 0.2042 | **0.1633** 🥈 |
| **Multi-Class Brier Score** | 0.2147 | 0.3579 | **0.1757** 🥇 | 0.1899 🥈 |

* **Pathological Asymmetry:** **Stem Bleeding** (trunk pathology) achieved near-perfect accuracy across all architectures (e.g., 2,406 / 2,420 correct on EfficientNet-B0, $99.4\%$). In contrast, over $72\%$ of all diagnostic confusion occurred mutually between **Gray Leaf Spot** and **Leaf Rot**, which share overlapping foliar necrotic symptoms.

---

### 2. Robustness Under Environmental Noise (12 Corruptions)

Images were perturbed across 4 corruption families at 3 severity levels:
- **Gaussian Defocus Blur** ($\sigma \in \{1.0, 2.0, 4.0\}$)
- **Extreme Darkness** ($0.8\times, 0.5\times, 0.2\times$)
- **Extreme Brightness / Glare** ($1.2\times, 1.8\times, 2.2\times$)
- **Cellular Compression (JPEG)** ($Q \in \{50, 20, 5\}$)

| Stress Regime | MobileNetV2 | ResNet50 | EfficientNet-B0 | DenseNet-121 |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Baseline** | 0.8844 | 0.8089 | **0.9328** | 0.9050 |
| **Blur 1 / Blur 2 / Blur 3** | 0.882 / 0.868 / 0.817 | 0.805 / 0.813 / 0.780 | 0.930 / 0.912 / 0.821 | **0.910 / 0.908 / 0.864** 🥇 |
| **Dark 1 / Dark 2 / Dark 3** | 0.878 / 0.891 / **0.881** 🥇 | 0.809 / 0.822 / 0.757 | **0.928 / 0.914** / 0.770 | 0.907 / 0.907 / 0.857 |
| **Bright 1 / Bright 2 / Bright 3** | 0.888 / 0.859 / 0.830 | 0.803 / 0.754 / 0.720 | **0.935 / 0.915 / 0.890** 🥇 | 0.901 / 0.874 / 0.844 |
| **JPEG 1 / JPEG 2 / JPEG 3** | 0.874 / 0.862 / 0.770 | 0.814 / 0.806 / 0.786 | **0.934 / 0.926 / 0.834** | 0.900 / 0.902 / **0.878** 🥇 |
| **Mean Robustness (12 Conditions)** | **0.8582** | 0.7890 | **0.8924** 🥇 | **0.8875** 🥈 |

* **Cliff vs. Slope:** Performance remains remarkably stable under lighting and JPEG noise (gentle slope), but experiences a sharp non-linear drop under heavy blur (cliff). **DenseNet-121 is the most resilient under severe blur ($0.8638$ Macro-F1)** due to dense feature reuse.
* **The "Sink Class" Discovery:** Under heavy blur ($\sigma = 4.0$), predictions do not become random; they collapse into majority foliar attractors (**Gray Leaf Spot absorbs 45%–50% of all predictions**). The sink class is an architectural artifact, not an inherent disease trait.

---

### 3. Probabilistic Calibration & Uncertainty Rejection (Selective Classification)

In field deployment, prompting a farmer to *"Retake Photo"* when confidence is low prevents costly agricultural misdiagnoses:

| Coverage Threshold | MobileNetV2 Accuracy | ResNet50 Accuracy | EfficientNet-B0 Accuracy | DenseNet-121 Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **100% Coverage (All images)** | 88.35% | 81.27% | **92.82%** | 89.95% |
| **90% Coverage (Reject bottom 10%)** | 92.40% | 86.15% | **95.60%** | 93.80% |
| **80% Coverage (Reject bottom 20%)** | **95.80%** | 90.45% | **97.90%** 🥇 | **96.50%** 🥈 |
| **60% Coverage** | 98.40% | 95.10% | **99.40%** | 98.90% |
| **50% Coverage (High Confidence Only)** | **99.20%** | 97.60% | **99.85%** 🥇 | **99.60%** 🥈 |

* **The 80% Sweet Spot:** Rejecting the 20% most uncertain samples pushes **EfficientNet-B0 to 97.90% accuracy** and **DenseNet-121 to 96.50%**.

---

### 4. Explainable AI (Grad-CAM Saliency & Stability)

Grad-CAM was evaluated to ensure models make decisions using true pathological symptoms rather than background soil, grass, or sky:
* **Visual Saliency:** Across all 5 categories, heatmaps accurately isolate the central growing bud for *Bud Rot*, nut attachment joints for *Bud Root Dropping*, and weeping trunk bark fissures for *Stem Bleeding*.
* **Attention Stability:** Under progressive noise, **DenseNet-121 retains the highest visual attention stability (mean Pearson $r = 0.7641$)**, followed by MobileNetV2 ($r = 0.6866$).
* **Failure Detection:** Using Grad-CAM geometric flip-consistency to identify misclassifications yields only $\text{AUROC} \approx 0.45\text{--}0.65$, whereas maximum softmax confidence ($\max p_i$) achieves $\text{AUROC} \approx 0.81\text{--}0.90$.

---

## Diagnostic Figures Gallery

All figures are automatically generated by the empirical evaluation suite into [`outputs/plots/report/`](outputs/plots/report/):

| Figure | Description | File Path |
| :---: | :--- | :--- |
| **Figure 1** | Clean Confusion Matrices across all 4 candidate models | [`outputs/plots/report/fig1_clean_confusion_matrices.png`](outputs/plots/report/fig1_clean_confusion_matrices.png) |
| **Figure 2** | Macro-F1 Robustness Curves across 12 corruption conditions | [`outputs/plots/report/fig2_robustness_curves.png`](outputs/plots/report/fig2_robustness_curves.png) |
| **Figure 3** | Side-by-side Confusion Matrices under severe `blur_3` and `jpeg_3` | [`outputs/plots/report/fig3_confusion_matrices_corruptions.png`](outputs/plots/report/fig3_confusion_matrices_corruptions.png) |
| **Figure 4** | Per-Class Recall Drop Heatmap under Severity 3 Noise | [`outputs/plots/report/fig4_vulnerability_heatmap.png`](outputs/plots/report/fig4_vulnerability_heatmap.png) |
| **Figure 5** | Accuracy on Answered Images vs. Coverage across 4 severe corruptions | [`outputs/plots/report/fig5_coverage_corruptions.png`](outputs/plots/report/fig5_coverage_corruptions.png) |
| **Figure 6** | Reliability Diagrams (Raw Softmax vs. $T_{\text{mixed}}$ Calibration) | [`outputs/plots/report/fig6_reliability_diagrams.png`](outputs/plots/report/fig6_reliability_diagrams.png) |
| **Figure 7** | Grad-CAM Attention Stability Bar Chart under Corruptions | [`outputs/plots/report/fig7_cam_stability_bars.png`](outputs/plots/report/fig7_cam_stability_bars.png) |
| **Figure 8** | Multi-Model Clean Grad-CAM Visual Heatmap Grid across all 5 classes | [`outputs/plots/report/fig8_clean_cam_grid.png`](outputs/plots/report/fig8_clean_cam_grid.png) |
| **Figure 9** | Progressive Visual Noise Breakdown on Palm Trunk (`Clean` $\to$ `Dark 3`) | [`outputs/plots/report/fig9_cam_corruption_breakdown.png`](outputs/plots/report/fig9_cam_corruption_breakdown.png) |

---

## Recommended Deployment Pipeline

Based on the empirical findings, we recommend the following 4-stage agricultural diagnosis pipeline:

```mermaid
graph TD
    A["Farmer Photo (Smartphone / Drone)"] --> B["Stage 1: Hardware Pre-Filter (Blur & Exposure Check)"]
    B -->|Blur Detected: Var < 100| C["User Prompt: 'Image too blurry, please retake'"]
    B -->|Passed Optical Gate| D["Stage 2: Neural Inference Engine"]
    D -->|Mobile / Edge (Offline)| E["MobileNetV2 (8.5 MB, 2.23M params, 88.35% Acc)"]
    D -->|Cloud / Server (Online)| F["EfficientNet-B0 (92.82% Acc) or DenseNet-121"]
    E --> G["Stage 3: Confidence Gating"]
    F --> G
    G -->|Max Softmax < 0.70| H["Flag as 'Uncertain: Agronomist Review Recommended'"]
    G -->|Max Softmax >= 0.70| I["Stage 4: Diagnosis & Grad-CAM Visual Overlay"]
    I --> J["Farmer UI with Saliency Heatmap"]
```

---

## Streamlined Repository Structure

```text
c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/
│
├── train.py                                  # CLI training pipeline with Two-Stage warmup and early stopping
├── evaluate.py                               # CLI evaluation runner & benchmark leaderboard generator
├── predict.py                                # CLI inference & Grad-CAM single-image prediction
├── requirements.txt                          # Python dependencies with PyTorch CUDA acceleration
├── README.md                                 # Primary repository landing page and documentation hub
│
├── src/                                      # Core Deep Learning Package
│   ├── config.py                             # Centralized hyperparameters, classes, and filesystem paths
│   ├── dataset.py                            # PyTorch Dataset, stratified splitting, and WeightedRandomSampler
│   ├── models.py                             # Model builder for MobileNetV2, ResNet50, EfficientNet-B0, DenseNet-121
│   ├── loss.py                               # Cost-sensitive loss implementations (Weighted CE, Focal Loss)
│   ├── metrics.py                            # Imbalance-aware metrics (Macro/Weighted F1, Recall, Precision)
│   └── explainability.py                     # Standalone Grad-CAM heatmap generator for single/multi-class images
│
├── scripts/                                  # Data Ingestion & Partitioning Utilities
│   ├── download_dataset.py                   # Kaggle dataset downloader utility
│   └── prepare_data.py                       # Dataset audit and stratified partitioning script
│
├── myDoc/                                    # Academic Documentation & Research Reports
│   ├── CocoGuard_Comprehensive_Research_Report.md  # Master empirical report (6 sections, full data & tables)
│   ├── CocoGuard_Research_Report.pdf               # Publication-quality 8-page compiled PDF report (11.9 MB)
│   ├── research_report_models_and_gradcam.md       # Technical report matching the PDF exhibits
│   ├── presentation_context_btp.md                 # Context document for generating BTP presentation slides
│   ├── data_imbalance_mitigation.md                # Mathematical framework for overcoming class imbalance
│   ├── train_explanation.md                        # Two-Stage training pipeline explanation
│   ├── project_progress_and_current_status.md      # Development milestone log
│   ├── progress.md                                 # Technical transition & operational log
│   └── build_pdf_report.py                         # Headless Chromium PDF compiler
│
├── Test and Eval/                            # Empirical Evaluation Suite & Stress-Testing Engine
│   ├── run_experiments.py                   # Automated overnight benchmark runner across 13,836 field images
│   ├── generate_empirical_figures.py         # Figure 3, 5, 6 generator & PDF rebuild pipeline
│   ├── generate_plots.py                     # Figure 1, 2, 4, 7 generator from experiment metrics
│   ├── generate_cam_figures.py               # Figure 8 & 9 high-res Grad-CAM grid generator
│   ├── corruptions.py                        # In-memory image corruption suite (Blur, Dark, Bright, JPEG)
│   ├── metrics_engine.py                     # Bootstrap CIs, ECE calibration, and TemperatureScaler
│   └── cam_engine.py                         # PyTorch Grad-CAM engine with Pearson, SSIM, and IoU metrics
│
├── data/
│   ├── raw/                                  # 5,798 raw training photos by class folder [Git-Ignored]
│   ├── splits/                               # Stratified train.csv, val.csv, and test.csv partitions
│   └── Coconut_Tree_Disease_Dataset.v3i.folder/ # 13,836 independent field images [Git-Ignored]
│
└── outputs/
    ├── checkpoints/                          # Saved model weights (*.pth, 8.5MB to 89MB) [Git-Ignored]
    ├── experiments/                          # experiment_results.json & empirical_logits.npz [Git-Ignored]
    ├── metrics/                              # Evaluation JSON logs and comparative CSV rankings
    ├── reports/                              # Generated PDF reports (.gitkeep tracked)
    └── plots/                                # Diagnostic figures
        └── report/                           # All 9 publication figures (fig1 through fig9)
```

---

## Installation & Quickstart

### 1. Environment Setup
```powershell
# Clone repository
git clone https://github.com/YourUsername/Coconut-Tree-Disease-Detection.git
cd Coconut-Tree-Disease-Detection

# Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Verify CUDA GPU detection:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```

### 2. Training Models
```powershell
# Train MobileNetV2 with Two-Stage Fine-Tuning and WeightedRandomSampler
python train.py --model mobilenet_v2 --epochs 25

# Train EfficientNet-B0 with Cost-Sensitive Weighted Cross-Entropy
python train.py --model efficientnet_b0 --loss weighted_ce --epochs 25

# Train DenseNet-121
python train.py --model densenet121 --epochs 25
```

### 3. Evaluating Models
```powershell
# Standard test split evaluation across all models
python evaluate.py --all

# Generate single-image Grad-CAM explanation
python explainability.py --model mobilenet_v2 --image path/to/leaf.jpg
```

### 4. Running the Empirical Robustness & Figure Suite
To evaluate all models on the 13,836 field dataset and generate Figures 3, 5, and 6:
```powershell
# Fast essential evaluation (~3.5 minutes on RTX 2050 GPU)
python "Test and Eval/generate_empirical_figures.py" --batch-size 128

# Re-compile the publication PDF report at any time
python myDoc/build_pdf_report.py
```

---

## References & Academic Citations

1. **Dataset Benchmark**: Thite, S., et al. (2023). *Multi-Class Coconut Tree Disease Dataset*.
2. **Grad-CAM**: Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. IEEE ICCV.
3. **MobileNetV2**: Sandler, M., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. IEEE CVPR.
4. **EfficientNet**: Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML.
5. **DenseNet**: Huang, G., et al. (2017). *Densely Connected Convolutional Networks*. IEEE CVPR.
6. **Calibration**: Guo, C., et al. (2017). *On Calibration of Modern Neural Networks*. ICML.

---

## Acknowledgments

We express our sincere gratitude to our project supervisors, faculty evaluators, and the **Department of Computer Science & Engineering at IIIT Kottayam** for their academic guidance, computational resources, and support throughout our Final Year B.Tech Project.
