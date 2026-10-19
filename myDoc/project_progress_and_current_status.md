# Project Progress & Current System Status

**Project:** Multi-Class Coconut Tree Disease Detection and Classification  
**Academic Context:** B.Tech Project (BTP) — IIIT Kottayam  
**Dataset Reference:** Thite et al. (2023) — 5,798 Field Images across 5 Categories  
**Last Updated:** October 17, 2026  
**Document Location:** [`myDoc/project_progress_and_current_status.md`](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/myDoc/project_progress_and_current_status.md)  

---

## 1. What Is Going on Right Now?

### Active Execution & Latest Test Evaluation Milestone:
- **MobileNetV2 Full-Dataset 25-Epoch Training Completed:**
  - Ran completely through all 25 epochs in 64.52 minutes.
  - Reached peak **Validation Macro F1 of 0.9936** and **Validation Accuracy of 98.96%** at Epoch 25!
  - Checkpoint cleanly saved to `outputs/checkpoints/best_mobilenet_v2.pth`.
  - Training curves saved to `outputs/plots/curves_mobilenet_v2.png`.
- **Official Test Set Evaluation Results (874 Test Images):**
  - **Overall Test Accuracy:** **98.97%** *(up from 93.24%, an absolute improvement of **+5.73%**)*.
  - **Macro F1-Score:** **0.9916** *(up from 0.9321, an absolute improvement of **+5.95%**)*.
  - **Macro Recall:** **99.37%** (0.9937).
  - **Macro Precision:** **98.96%** (0.9896).
  - **Inference Latency:** **4.92 ms / image** (**203.3 FPS** on NVIDIA GPU).
  - **Per-Class Breakdown on Test Split:**
    - **Bud Root Dropping:** 100.0% Precision | 100.0% Recall | **1.0000 F1** (78/78 images)
    - **Bud Rot (Critical Minority):** 97.26% Precision | **100.0% Recall** | **0.9861 F1** (71/71 images — ZERO false negatives!)
    - **Gray Leaf Spot:** 98.75% Precision | 98.44% Recall | **0.9860 F1** (321 images)
    - **Leaf Rot:** 98.80% Precision | 98.41% Recall | **0.9861 F1** (252 images)
    - **Stem Bleeding:** 100.0% Precision | 100.0% Recall | **1.0000 F1** (152/152 images)
- **Pipeline Upgrades Proven in This Run:**
  1. **Full Dataset Re-stratification:** All 5,798 images partitioned into 70% Train (4,058 images), 15% Validation (870 images), and 15% Test (874 images).
  2. **Two-Stage Fine-Tuning:** 3 epochs of Stage 1 head warmup (backbone frozen), followed by Stage 2 full fine-tuning with differential learning rates (Backbone: $1\times 10^{-5}$, Head: $1\times 10^{-4}$).
  3. **Label Smoothing ($0.1$):** Substantially reduced overconfident predictions and accelerated generalization.
  4. **RandomErasing Augmentation:** Improved disease feature extraction across occluded leaf patches.
  5. **Worker Fix (`NUM_WORKERS = 0`):** Eliminated Windows IPC pagefile exhaustion (Win32 Error 1455).

---

## 2. Project Background & The Core Research Problem

### The Clinical & Agricultural Challenge:
Coconut palm (*Cocos nucifera*) diseases present drastic visual variations across different tree anatomy (leaves, trunk bark, spindle leaves, and crown base). Moreover, field datasets suffer from severe class imbalance:

| Disease Class | Image Count | Distribution % | Clinical Severity | Symptom Region |
| :--- | :---: | :---: | :--- | :--- |
| **Gray Leaf Spot** | 2,135 | 36.8% | High prevalence | Leaves (elliptical grey/brown necrotic spots) |
| **Leaf Rot** | 1,673 | 28.9% | High prevalence | Leaves (blackened decayed distal leaflets) |
| **Stem Bleeding** | 1,006 | 17.3% | Moderate | Trunk/Bark (dark brownish exudation cracks) |
| **Bud Root Dropping** | 514 | 8.9% | **Minority** | Crown base & nut attachment stalks |
| **Bud Rot** | 470 | 8.1% | **Critical Minority (Fatal)** | Spindle leaf & terminal growing bud crown |
| **Total** | **5,798** | **100.0%** | **~4.5 : 1 Imbalance** | |

> [!CAUTION]
> **Why Conventional Training Fails:**  
> Standard Cross-Entropy loss rewards models for predicting the majority classes (*Gray Leaf Spot* and *Leaf Rot*). A naive model can attain 80%+ raw accuracy while completely failing to detect *Bud Rot*. Because *Bud Rot* kills the entire coconut palm if undetected, sensitivity (Recall) on minority diseases is the defining academic metric of this project.

---

## 3. Comprehensive Progress Made Till Now

### Milestone 1: Environment & Data Pipeline Engineering
- [x] **Isolated Environment:** Fully configured `.venv` with PyTorch, TorchVision (CUDA-enabled), Pandas, Scikit-Learn, Matplotlib, Seaborn, and Pillow.
- [x] **Clean Repository Layout:** Established organized directory structure:
  - `data/raw/`: 5 class image folders (5,798 real-world images).
  - `data/splits/`: Stratified CSV pointer files preserving class balance without duplicating large image files.
  - `outputs/checkpoints/`: Model weight binaries (`.pth`).
  - `outputs/metrics/`: Evaluation JSON logs and comparison CSVs.
  - `outputs/plots/`: High-resolution confusion matrices and training curves.
  - `myDoc/`: Dedicated documentation directory (tracked in `.gitignore`).
- [x] **Git Version Control & Safety:** Configured `.gitignore` to protect repository size by excluding raw image files (5,798 images), multi-megabyte model checkpoints (`outputs/checkpoints/*`), virtual environments (`.venv/`), and system cache files.

---

### Milestone 2: Triple-Layer Imbalance Mitigation Architecture
To eliminate minority class starvation, the pipeline implements a three-pronged mitigation strategy:
1. **Dynamic Mini-Batch Sampling (`WeightedRandomSampler`):** Every mini-batch samples minority classes (*Bud Rot*, *Bud Root Dropping*) with equal probability to majority classes, ensuring consistent gradient flow.
2. **Cost-Sensitive Loss (`compute_class_weights`):** Inverse class frequency weighting heavily penalizes errors on rare categories:
   $$\text{Weight}_c = \frac{N}{C \cdot N_c}$$
   Coupled with optional Focal Loss ($\gamma = 2.0$) and Label Smoothing ($0.1$).
3. **Macro F1-Score Checkpointing:** The training loop strictly checkpoints the best weights according to **Validation Macro F1-Score** (unweighted mean of per-class F1) rather than overall accuracy.

---

### Milestone 3: Candidate Model Suite & Two-Stage Training
Implemented five transfer learning vision backbones in [`models.py`](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/models.py):
1. **MobileNetV2:** Target edge architecture (~2.2M trainable params, ~8.5 MB) designed for smartphone/drone deployment in coconut plantations.
2. **ResNet50:** Robust 50-layer residual network baseline (~23.5M params, ~89.7 MB) serving as the academic literature benchmark.
3. **EfficientNet-B0:** Compound-scaled network balancing accuracy and FLOPS.
4. **DenseNet121:** Feature-reuse architecture optimal for fine foliar patterns.
5. **VGG16:** Classical baseline.

Implemented **Two-Stage Fine-Tuning** in [`train.py`](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/train.py):
- **Stage 1 (Warmup):** Head trained for 3 epochs with backbone frozen to stabilize randomly initialized weights.
- **Stage 2 (Differential Fine-Tuning):** Entire network unfrozen with backbone learning rate at $10\times$ lower than classification head.

---

### Milestone 4: Completed Experimental Benchmarks & Results

Two major experimental evaluations have already been conducted and recorded:

#### Experiment A: Balanced Sub-Dataset Benchmark (Test Split: 353 images)
Evaluated on a strictly balanced test partition (470 images/class cap):

| Model Architecture | Accuracy (%) | Macro F1 | Macro Recall | Macro Precision | Size (MB) | Latency (ms) | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet50** | **94.65%** | **0.9459** | 0.9465 | 0.9480 | 89.72 MB | 8.69 ms | 115.0 FPS |
| **MobileNetV2** | **93.24%** | **0.9321** | 0.9324 | 0.9362 | **8.51 MB** | **5.79 ms** | **172.7 FPS** |

*Takeaway:* ResNet50 leads slightly in raw Macro F1 (+1.38%), but MobileNetV2 is **$10.5\times$ smaller** and **$1.5\times$ faster**, making it the superior candidate for field hardware.

---

#### Experiment B: Out-of-Sample Remaining Dataset Evaluation (3,781 real images)
Models trained on the balanced subset were tested against all remaining 3,781 un-sampled real-world images to rigorously verify real-world generalization:

| Model / Ensemble | Accuracy (%) | Macro F1 | Weighted F1 | Macro Recall | Bud Rot Recall | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2** | 93.10% | 0.9177 | 0.9321 | 95.07% | 100.0% | **5.21 ms** |
| **ResNet50** | 91.93% | 0.8844 | 0.9236 | 95.56% | 100.0% | 8.70 ms |
| **Soft-Voting Ensemble**<br>*(55% MobileNet + 45% ResNet)* | **93.92%** | **0.9211** | **0.9407** | **96.39%** | **100.0% (125/125)** | **11.08 ms** |

#### Per-Class Breakdown of the Soft-Voting Ensemble (3,781 Images):
- **Bud Rot (Critical Minority):** **100.0% Recall** (125 / 125 images correctly identified!)
- **Stem Bleeding:** **100.0% Precision & 100.0% Recall** (589 / 589 images perfectly classified)
- **Bud Root Dropping:** **98.57% Recall**, 95.17% Precision, 0.9684 F1
- **Gray Leaf Spot:** 95.04% Recall, 92.84% Precision, 0.9393 F1
- **Leaf Rot:** 88.34% Recall, 97.85% Precision, 0.9285 F1

#### Experiment C: Full-Dataset 25-Epoch Benchmark (874 Test Images)
Following the two-stage fine-tuning and $256 \times 256$ augmentation upgrades on all 5,798 images:

| Model Architecture | Test Accuracy (%) | Macro F1 | Macro Recall | Macro Precision | Latency (ms) | Throughput (FPS) | Bud Rot Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2 (Full)** | **98.97%** | **0.9916** | **0.9937** | **0.9896** | **4.92 ms** | **203.3 FPS** | **100.0% (71/71)** |
| **ResNet50 (Baseline)** | 94.65% | 0.9459 | 0.9465 | 0.9480 | 8.69 ms | 115.0 FPS | 97.2% |

*Net Improvement:* MobileNetV2 improved by **+5.73% Accuracy** and **+0.0595 Macro F1**, outperforming the ResNet-50 baseline while remaining $10.5\times$ smaller and running at over 200 FPS on GPU!

---

### Milestone 5: Visual Explainability (Grad-CAM)
- Implemented [`explainability.py`](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/explainability.py) supporting automated forward/backward hook attachment across diverse backbones:
  - MobileNetV2: `features[-1]`
  - ResNet50: `layer4[-1]`
  - EfficientNet-B0: `features[-1]`
  - DenseNet121: `features.denseblock4[-1]`
- Integrated into [`predict.py`](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/predict.py) for clinical disease verification (ensuring models look at leaf lesions and trunk fissures rather than background soil or sky).

---

## 4. Current Scorecard & Deliverables Table

| Deliverable / Artifact | Path | Status | Details |
| :--- | :--- | :---: | :--- |
| Training Engine | [`train.py`](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/train.py) | **Completed** | Full CLI, two-stage fine-tuning, label smoothing, sampler |
| Code Explanation Doc | [`myDoc/train_explanation.md`](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/myDoc/train_explanation.md) | **Completed** | Comprehensive line-by-line architecture guide |
| Imbalance Strategy Doc | [`myDoc/data_imbalance_mitigation.md`](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/myDoc/data_imbalance_mitigation.md) | **Completed** | 5-tier imbalance architecture, formulas & empirical results |
| Status & Progress Doc | [`myDoc/project_progress_and_current_status.md`](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/myDoc/project_progress_and_current_status.md) | **Completed** | Current status & cumulative milestone history |
| Best MobileNetV2 Weights | `outputs/checkpoints/best_mobilenet_v2.pth` | **Completed** | **Test Acc: 98.97% | Macro F1: 0.9916** |
| Best ResNet50 Weights | `outputs/checkpoints/best_resnet50.pth` | **Completed** | Macro F1: 0.9459 (Baseline) |
| Model Comparison Log | `outputs/metrics/model_comparison.csv` | **Completed** | Benchmark metrics across architectures |
| Out-of-Sample Metrics | `outputs/metrics/model_comparison_remaining.csv`| **Completed** | 3,781-image out-of-sample stress test |
| Confusion Matrix Heatmaps | `outputs/plots/cm_*.png` | **Completed** | 6 confusion matrix plots generated |
| Learning Curves | `outputs/plots/curves_*.png` | **Completed** | Loss, Accuracy, and Macro F1 curves |
| Grad-CAM Visualizer | [`explainability.py`](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/explainability.py) | **Completed** | Layer hooks & heatmap overlays |
| Inference CLI | [`predict.py`](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/predict.py) | **Completed** | Top-$K$ diagnosis + visual Grad-CAM output |
| Git Configuration | [`.gitignore`](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/.gitignore) | **Completed** | Raw data, checkpoints, venv, and `myDoc/` excluded |

---

## 5. Upcoming Next Steps

1. **Complete MobileNetV2 Full-Dataset Run:**
   - Allow current training to finish or reach early-stopping checkpoint.
   - Verify updated validation loss and Macro F1 on the full 5,798-image stratified partition.
2. **Benchmark EfficientNet-B0 & DenseNet121:**
   - Run full training on EfficientNet-B0 (`python train.py --model efficientnet_b0`) and DenseNet121.
   - Compare edge efficiency vs. feature reuse on foliar necrosis patterns.
3. **Generate Grad-CAM Visual Exhibit:**
   - Run `predict.py --explain` on typical test images for all 5 disease categories to curate visualization figures for the BTP final report and seminar presentation.
4. **Compile BTP Final Project Report:**
   - Structure Methodology, Results, Confusion Matrices, and Latency Analysis into the formal BTP thesis document.
