# BTP Presentation Master Context & Slide Deck Blueprint
**Multi-Class Coconut Tree Disease Detection & Classification with Class-Imbalance Mitigation**

> **Notice for Gemini / LLM Slide Generator:**  
> Use the structured information, empirical tables, architectural details, and slide-by-slide outline below to generate a publication-quality, academic presentation (PPT) for the B.Tech Capstone Project Review at **IIIT Kottayam**.  
> **Scope Boundary:** This context encompasses all research, methodology, systems engineering, model training, and comparative evaluation up to the official evaluation of all four candidate models (`evaluate.py --all`).

---

## 1. Title & Team Information

- **Project Title:** Multi-Class Coconut Tree Disease Detection and Classification using Deep Learning with Class-Imbalance Mitigation
- **Institution:** Department of Computer Science and Engineering, Indian Institute of Information Technology (IIIT) Kottayam
- **Academic Program:** Bachelor of Technology (B.Tech) — Final Year Capstone Project (BTP)
- **Project Team:**
  1. **Ansh Singh** — `2023BCS0120`
  2. **Lakshmi Harshitha Cheedella** — `2023BCS0069`
  3. **Soumallya Sarkar** — `2023BCS0172`
  4. **Chennupati Bhargav** — `2023BCS0219`

---

## 2. Project Motivation & Agricultural Domain Problem

### The Clinical Problem
Coconut palm (*Cocos nucifera*) is an essential cash crop supporting millions of farming families across coastal India and tropical nations. However, coconut plantations suffer from lethal and debilitating fungal, bacterial, and physiological diseases:

1. **Bud Rot (*Phytophthora palmivora*):** The most devastating, lethal infection. It attacks the growing spindle leaf and central vegetative bud. If undetected, **it kills the entire palm within weeks**. Early detection is an urgent matter of survival for the tree.
2. **Bud Root Dropping:** Attacks fruit stalks and inflorescence, causing immature button nuts to drop, slashing agricultural yield by 40–70%.
3. **Gray Leaf Spot (*Pestalotiopsis palmarum*):** Foliar fungal disease causing severe necrotic lesions on leaves, reducing photosynthetic capacity.
4. **Leaf Rot (*Bipolaris incurvata*):** Decays and blackens distal leaflets, weakening frond structural integrity.
5. **Stem Bleeding (*Thielaviopsis paradoxa*):** Trunk pathology producing reddish-brown exudation through bark fissures, leading to internal vascular decay.

### The Problem with Manual Diagnosis
- Traditional manual inspection requires agricultural experts to physically survey vast acreage.
- Symptoms on tall tree crowns (15–30 meters high) are easily overlooked until the tree is already dead.
- Visual similarity between foliar diseases (*Gray Leaf Spot* vs. *Leaf Rot*) leads to misdiagnosis and incorrect fungicide application.

---

## 3. The Core Machine Learning Challenge: Severe Real-World Class Imbalance

Our study evaluates the standard benchmark field dataset (Patil, Thite et al., 2023), comprising **5,798 real-world plantation images**:

| Class Index | Disease Category | Image Count | Distribution (%) | Clinical Urgency | Pathology Target |
| :---: | :--- | :---: | :---: | :--- | :--- |
| `0` | **Bud Root Dropping** | 514 | **8.87%** | High | Nut stalks & crown base |
| `1` | **Bud Rot** | 470 | **8.11%** | **CRITICAL (Fatal)** | Terminal growing bud |
| `2` | **Gray Leaf Spot** | 2,135 | **36.82%** | Moderate | Foliar leaflets |
| `3` | **Leaf Rot** | 1,673 | **28.85%** | Moderate | Foliar leaflets |
| `4` | **Stem Bleeding** | 1,006 | **17.35%** | High | Trunk vascular tissue |
| — | **Total Dataset** | **5,798** | **100.00%** | **~4.5 : 1 Imbalance Ratio** | |

### Why Conventional Deep Learning Fails Here:
- **Majority Class Domination:** Over **65%** of the dataset consists of leaf spots (*Gray Leaf Spot* and *Leaf Rot*).
- **The "Accuracy Paradox":** A standard Convolutional Neural Network (CNN) trained with standard Cross-Entropy can simply ignore minority classes (*Bud Rot* at 8.1%) and still report **>85% overall accuracy** while failing to detect the single disease that actually kills the tree.
- **Sensitivity on Bud Rot is paramount:** In agricultural pathology, a false negative on *Bud Rot* is catastrophic.

---

## 4. Proposed 5-Pillar Imbalance Mitigation Architecture

To completely eliminate minority-class starvation, our team engineered a unified, multi-tier mitigation pipeline:

```
┌────────────────────────────────────────────────────────────────────────┐
│              TRIPLE-TIER IMBALANCE MITIGATION FRAMEWORK                │
├──────────────────────────┬─────────────────────────────────────────────┤
│ 1. Data-Level Rebalancing│ Dynamic WeightedRandomSampler (20% quota)   │
│ 2. Cost-Sensitive Loss   │ Inverse-Frequency Weighted Cross-Entropy    │
│ 3. Regularization        │ Label Smoothing (ε = 0.1)                   │
│ 4. Two-Stage Fine-Tuning │ Stage 1 Warmup (3 epochs) -> Stage 2 Diff LR│
│ 5. Objective Metric      │ Checkpointing strictly on Macro F1-Score    │
└──────────────────────────┴─────────────────────────────────────────────┘
```

1. **Stratified Data Splitting (70% / 15% / 15%):**
   - Partitioned the 5,798 images into **Train (4,058 images)**, **Validation (870 images)**, and **Test (874 images)**, strictly preserving exact class proportions across all three splits.
2. **Dynamic Mini-Batch Sampling (`WeightedRandomSampler`):**
   - Sample weights are assigned inversely proportional to class frequency:
     $$w_i = \frac{1}{N_{c(i)}}$$
   - **Result:** Every mini-batch contains an approximately equal number of images (~20%) from each of the 5 classes. Minority classes are sampled with replacement, guaranteeing consistent gradient updates.
3. **Cost-Sensitive Weighted Cross-Entropy Loss:**
   - Multiplies the cross-entropy penalty by inverse-frequency weights:
     $$W_c = \frac{N_{\text{total}}}{C \cdot N_c}$$
   - Strongly penalizes misclassifications on rare diseases like *Bud Rot* ($W_{\text{Bud Rot}} \approx 2.47$) compared to majority classes ($W_{\text{Gray Leaf Spot}} \approx 0.54$).
4. **Label Smoothing Regularization ($\epsilon = 0.1$):**
   - Replaces hard one-hot targets with soft distributions ($y_k = 0.92$, non-target $y_j = 0.02$).
   - Prevents the network from becoming overconfident on majority classes and encourages tighter feature clustering.
5. **Macro F1-Score Checkpointing:**
   - Checkpointing monitors **Validation Macro F1** (the unweighted arithmetic mean of F1 across all 5 classes), preventing majority classes from masking minority errors:
     $$\text{Macro F1} = \frac{1}{5} \sum_{c=0}^{4} \text{F1}_c$$

---

## 5. Candidate Deep Learning Architectures

We evaluated four diverse, state-of-the-art transfer learning architectures to identify the optimal balance between diagnostic accuracy, computational complexity, and edge deployability:

| Architecture | Paradigm / Key Innovation | Parameters | Model Size | Target Deployment Profile |
| :--- | :--- | :---: | :---: | :--- |
| **MobileNetV2** | Inverted Residuals & Linear Bottlenecks | **2.23 M** | **8.51 MB** | **Ultra-lightweight edge hardware, drones, Android/iOS apps** |
| **ResNet50** | Residual Shortcut Connections (Skip-connections) | 23.52 M | 89.72 MB | Standard academic baseline for deep visual representations |
| **EfficientNet-B0** | Neural Architecture Search (NAS) & Compound Scaling | 4.01 M | 15.31 MB | High-efficiency edge servers and low-power cloud backends |
| **DenseNet-121** | Dense Feature Reuse (Direct connections across layers) | 6.96 M | 26.55 MB | Maximum feature preservation for fine-grained leaf necrosis |

---

## 6. Training Strategy & Systems Engineering

### Two-Stage Fine-Tuning Protocol
- **Stage 1: Head Warmup (Epochs 1–3):**
  - Convolutional backbone weights are frozen.
  - Only the newly initialized linear classification head is trained at $\text{LR} = 10^{-4}$ with AdamW.
  - *Rationale:* Prevents large, chaotic initial gradients from destroying pre-trained ImageNet feature extractors.
- **Stage 2: Differential Learning Rate Fine-Tuning (Epochs 4–25):**
  - The entire network is unfrozen.
  - Backbone trains at a conservative learning rate ($\text{LR} = 10^{-5}$) while the classification head trains at $\text{LR} = 10^{-4}$.
  - Uses `CosineAnnealingLR` with early stopping (patience: 7 epochs).

### Systems & Performance Optimization
- **Hardware:** Accelerated on an **NVIDIA GeForce RTX 2050 Laptop GPU (4.0 GB VRAM)** with CUDA 12.1.
- **Resolving Windows IPC Bottleneck:**
  - Diagnostic profiling identified that multi-process PyTorch dataloading (`num_workers > 0`) on Windows created severe IPC pipe serialization overhead (~2.40s per iteration).
  - Resolved by setting `NUM_WORKERS = 0` with pinned GPU memory and eliminating redundant interpolation steps in `dataset.py`.
  - **Throughput jumped from 0.4 it/s to 2.65 it/s (~6.6x acceleration)**, cutting training time from hours to ~18 minutes per run.

---

## 7. Official Comparative Benchmark Results (`evaluate.py --all`)

All four candidate architectures and a multi-model Soft-Voting Ensemble were evaluated on the **held-out test set (874 unseen images)** under identical, standardized conditions:

### Master Comparison Leaderboard:

| Rank | Model Architecture | Test Accuracy (%) | Macro F1 | Weighted F1 | Macro Recall | Macro Precision | Parameters | Storage Size | Latency (GPU) | Throughput (FPS) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **DenseNet-121** | **99.66%** | **0.9979** | **0.9966** | **0.9981** | **0.9976** | 6.96 M | 26.55 MB | 116.7 ms | 8.6 FPS |
| 🥈 | **EfficientNet-B0** | **99.43%** | **0.9965** | **0.9943** | **0.9965** | **0.9964** | 4.01 M | 15.31 MB | 26.2 ms | 38.2 FPS |
| 🥉 | **Soft-Voting Ensemble** | **99.31%** | **0.9957** | **0.9931** | **0.9957** | **0.9957** | 38.95 M | 148.60 MB | 183.3 ms | 5.5 FPS |
| 4 | **MobileNetV2** | **98.97%** | **0.9916** | **0.9897** | **0.9937** | **0.9896** | **2.23 M** | **8.51 MB** | **64.3 ms** | **15.5 FPS** |
| 5 | **ResNet50** | **93.14%** | **0.9327** | **0.9312** | **0.9446** | **0.9250** | 23.52 M | 89.72 MB | 180.8 ms | 5.5 FPS |

---

### Detailed Findings & Key Analytical Insights:

1. **DenseNet-121 Achieved State-of-the-Art Performance:**
   - Delivered **99.66% Accuracy** and **0.9979 Macro-F1**, misclassifying only 3 images out of 874.
   - Its dense connectivity enables rich feature reuse across spatial scales, uniquely suited to distinguishing fine fungal spot margins from broad leaflet decay.
2. **EfficientNet-B0 is the Best Accuracy-to-Weight Architecture:**
   - Achieved **99.43% Accuracy** and **0.9965 Macro-F1** with only **4.01M parameters (15.3 MB)**.
   - Achieved the highest throughput among the top models (**38.2 FPS**), making it the premier choice for real-time video processing.
3. **MobileNetV2 is the Ultimate Edge & Mobile Deployment Winner:**
   - Achieved **98.97% Accuracy** with an astonishingly compact footprint of just **2.23M parameters (8.51 MB)**.
   - Fits effortlessly on smartphones, embedded Raspberry Pi/Jetson modules, and drone payloads without requiring cloud connectivity.
4. **ResNet50 Underperformed Compound/Dense Architectures:**
   - Despite having the largest parameter count (23.5M), ResNet-50 scored 93.14%, proving that sheer model size is inferior to architectural efficiency (compound scaling in EfficientNet and dense feature reuse in DenseNet).
5. **Zero False Negatives on Critical Bud Rot:**
   - Both MobileNetV2 and DenseNet121 achieved **100.0% Recall on Bud Rot** (all infected palms correctly flagged). Our imbalance mitigation completely eliminated fatal diagnostic blind spots.

---

## 8. Explainability via Grad-CAM (Gradient-Weighted Class Activation Mapping)

To verify that the models make decisions based on true botanical pathology rather than background shortcuts (such as soil, sky, or plantation fences), we implemented **Grad-CAM**:

- **Target Layers Hooked:**
  - `MobileNetV2`: `model.features[-1]` (last 1280-dim bottleneck)
  - `ResNet50`: `model.layer4[-1]` (last residual block)
  - `EfficientNet-B0`: `model.features[-1]` (last compound block)
  - `DenseNet-121`: `model.features.denseblock4` (final dense feature extractor)
- **Visual Validation Insights:**
  - **Bud Rot:** Activation heatmaps focus tightly on the central vegetative crown and spindle leaf base.
  - **Gray Leaf Spot:** Activation heatmaps isolate individual elliptical necrotic patches while ignoring healthy green lamina.
  - **Stem Bleeding:** Activation maps pinpoint vertical dark resin exudations on the lower trunk bark.

---

## 9. Ready-to-Use 12-Slide Presentation Outline for Gemini

When generating your slide deck, use this clean 12-slide layout:

### Slide 1: Title & Team Credentials
- **Title:** Multi-Class Coconut Tree Disease Detection & Classification
- **Subtitle:** Addressing Real-World Class Imbalance with Deep Convolutional Ensembles
- **Department:** Department of CSE, IIIT Kottayam
- **Team Members:** Ansh Singh, Lakshmi Harshitha Cheedella, Soumallya Sarkar, Chennupati Bhargav

### Slide 2: The Agricultural Problem & Clinical Context
- Economic significance of *Cocos nucifera*.
- Overview of 5 pathologies: Bud Rot (fatal), Bud Root Dropping, Gray Leaf Spot, Leaf Rot, Stem Bleeding.
- The failure of manual diagnostic surveys (tall crowns, rapid pathogen spread).

### Slide 3: The Data Imbalance Challenge
- Dataset profile: 5,798 field images (Patil, Thite et al., 2023).
- 4.5:1 imbalance ratio (Gray Leaf Spot 36.8% vs. Bud Rot 8.1%).
- The Accuracy Paradox: why standard models fail on rare, fatal diseases.

### Slide 4: Proposed 5-Pillar Imbalance Mitigation
- Visual architecture diagram of the mitigation pipeline.
- Stratified 70/15/15 partitioning.
- Dynamic `WeightedRandomSampler` (equal 20% mini-batch quota).
- Cost-Sensitive Weighted Cross-Entropy Loss + Label Smoothing ($\epsilon = 0.1$).
- Macro F1-score validation monitoring.

### Slide 5: Candidate Vision Architectures
- Comparative table of the 4 architectures: MobileNetV2, ResNet50, EfficientNet-B0, DenseNet-121.
- Design rationale: Edge mobile efficiency vs. deep feature reuse.

### Slide 6: Two-Stage Transfer Learning Protocol
- Stage 1 Warmup (frozen backbone, head initialization).
- Stage 2 Fine-Tuning (differential learning rates, CosineAnnealing).
- Prevention of catastrophic forgetting.

### Slide 7: Systems Engineering & Hardware Acceleration
- Hardware environment: NVIDIA RTX 2050 (4 GB VRAM) CUDA acceleration.
- Diagnosing and resolving the Windows IPC worker bottleneck (`NUM_WORKERS = 0`).
- 6.6x training speedup (from 0.4 it/s to 2.65 it/s).

### Slide 8: Master Evaluation Leaderboard (`evaluate.py --all`)
- Benchmark table across all 4 models and the ensemble on 874 test images.
- Highlight DenseNet-121 (99.66%) and EfficientNet-B0 (99.43%).
- Highlight MobileNetV2 (98.97% in just 8.5 MB).

### Slide 9: Minority Class Protection & Per-Class Metrics
- Per-class breakdown: Bud Rot achieved **100.0% Recall** (zero missed infections).
- Analysis of Leaf Rot vs. Gray Leaf Spot boundary precision.
- Proof that cost-sensitive weighting protected minority classes.

### Slide 10: Multi-Model Soft-Voting Ensemble
- Methodology: Weighted combination of candidate probabilities.
- Achieving 99.31% Accuracy while reducing individual model variance.

### Slide 11: Explainable AI with Grad-CAM
- Gradient-weighted visual heatmaps on crown, fronds, and trunk.
- Proof that models attend to genuine botanical lesions rather than background soil/sky.

### Slide 12: Conclusions & Project Impact
- Key takeaways: Class imbalance resolved without synthetic data artifacts.
- DenseNet-121 leads in diagnostic precision; MobileNetV2 leads in edge deployability.
- Future deployment: Android mobile app and drone-mounted plantation surveying.
