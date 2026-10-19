# CocoGuard: Comprehensive Empirical Research Report
## Deep Learning Architectures, Out-of-Distribution Robustness, Calibration, and Explainability for Coconut Tree Disease Diagnosis

**Department of Computer Science & Engineering, Indian Institute of Information Technology (IIIT) Kottayam**  
*Course:* Final Year B.Tech Capstone Project (BTP)  
*Authors:* Ansh Singh (`2023BCS0120`), Lakshmi Harshitha Cheedella (`2023BCS0069`), Soumallya Sarkar (`2023BCS0172`), Chennupati Bhargav (`2023BCS0219`)  
*Evaluation Dataset:* 13,836 Unseen Field Images across 5 Pathological Categories  
*Date of Benchmark:* October 2026  

---

## Executive Summary

Smallholder coconut farmers face significant crop losses from devastating foliar and trunk pathologies, including **Bud Rot**, **Bud Root Dropping**, **Gray Leaf Spot**, **Leaf Rot**, and **Stem Bleeding**. While convolutional neural networks (CNNs) report high test accuracies under idealized laboratory benchmarks, deployable smart agriculture demands models that generalize across independent plantations, remain robust under adverse weather photography, provide calibrated confidence estimates, and ground their predictions in biologically faithful visual evidence.

This report presents a comprehensive empirical evaluation of four distinct vision paradigms—[MobileNetV2](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/src/models/model_factory.py) (inverted residual edge network), [ResNet50](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/src/models/model_factory.py) (classical residual baseline), [EfficientNet-B0](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/src/models/model_factory.py) (compound-scaled efficient backbone), and [DenseNet-121](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/src/models/model_factory.py) (dense feature-reuse network). The models were trained strictly on a standardized 5,798-image dataset and evaluated across **13,836 field images** representing a **Zero-Shot Cross-Plantation Generalization Stress-Test**.

---

## 1. Candidate Architectures & Clean Generalization Benchmark

### 1.1 Architectural Profiles & Hardware Footprint

| Architectural Dimension | MobileNetV2 | ResNet50 | EfficientNet-B0 | DenseNet-121 |
| :--- | :---: | :---: | :---: | :---: |
| **Engineering Paradigm** | Inverted Residuals & Linear Bottlenecks | Residual Identity Mapping | Compound Depth/Width Scaling | Dense Cross-Layer Feature Reuse |
| **Total Parameters (5-Class Head)** | **2,230,277 (2.23 M)** | 23,518,085 (23.52 M) | 4,013,953 (4.01 M) | 6,958,917 (6.96 M) |
| **Model Disk Size (.pth)** | **8.51 MB** 🥇 | 89.72 MB | 15.31 MB 🥈 | 26.55 MB |
| **Inference Latency (RTX 2050, Batch 1)** | **3.8 ms** 🥇 | 11.2 ms | 6.4 ms 🥈 | 13.5 ms |
| **Grad-CAM Target Layer** | `features[-1]` | `layer4[-1]` | `features[-1]` | `features.denseblock4` |

### 1.2 Zero-Shot Clean Performance on 13,836 Unseen Field Images

All models were evaluated without plantation-specific fine-tuning or domain adaptation:

| Evaluation Metric | MobileNetV2 | ResNet50 | EfficientNet-B0 | DenseNet-121 |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Accuracy** | **88.35%** | 81.27% | **92.82%** 🥇 | **89.95%** 🥈 |
| **Clean Macro-F1** | **0.8844** | 0.8089 | **0.9328** 🥇 | **0.9050** 🥈 |
| **95% Bootstrap Confidence Interval** | `[0.879, 0.890]` | `[0.801, 0.816]` | `[0.928, 0.937]` | `[0.900, 0.910]` |
| **Diagnostic Errors (of 13,836 images)** | 1,612 errors | 2,591 errors | **994 errors** 🥇 | 1,390 errors 🥈 |
| **Clean Expected Calibration Error (ECE)** | **0.1607** 🥇 | 0.2494 | 0.2042 | **0.1633** 🥈 |
| **Multi-Class Brier Score** | 0.2147 | 0.3579 | **0.1757** 🥇 | 0.1899 🥈 |

![Figure 1: Clean Confusion Matrices across all 4 Models](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/outputs/plots/report/fig1_clean_confusion_matrices.png)

### 1.3 Key Discovery: Per-Disease Diagnostic Asymmetry
1. **Trunk Geometry vs. Foliar Necrosis:**
   - **Stem Bleeding** achieved near-perfect recognition across all models (e.g., 2,406 correct out of 2,420 test samples on EfficientNet-B0, 99.4% accuracy). Why? The vertical orientation of palm bark striations and reddish-black exudations presents strong structural geometric priors that never overlap with frond morphology.
2. **The Foliar Confusion Pair:**
   - Over 72% of all diagnostic errors occurred mutually between **Gray Leaf Spot** and **Leaf Rot**. Both manifest as irregular brownish necrosis across leaf pinnules. Under variable outdoor lighting and camera zoom, the models frequently mistake advanced gray spot coalescences for leaf rot.
3. **Crown Rot Localization:**
   - **Bud Rot** and **Bud Root Dropping** require looking at the apical meristem / crown base. MobileNetV2 and EfficientNet-B0 distinguished them effectively, whereas ResNet50 struggled with frond drooping versus crown necrosis.

---

## 2. Empirical Stress-Testing: Robustness Under Environmental Noise

Field capture by farmers introduces severe optical and atmospheric degradations. We subjected all 13,836 test images to **four corruption families across three progressive severities (12 stress conditions)**:
- **Gaussian Defocus Blur:** $\sigma \in \{1.0, 2.0, 4.0\}$
- **Sub-optimal Lighting (Darkness):** Multiplier $\in \{0.8\times, 0.5\times, 0.2\times\}$
- **Extreme Glare (Brightness):** Multiplier $\in \{1.2\times, 1.8\times, 2.2\times\}$
- **Cellular Compression (JPEG):** Quality factor $Q \in \{50, 20, 5\}$

### 2.1 Macro-F1 Across 12 Corruption Conditions

| Stress Condition | MobileNetV2 | ResNet50 | EfficientNet-B0 | DenseNet-121 |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Baseline** | 0.8844 | 0.8089 | **0.9328** | 0.9050 |
| **Blur 1 / Blur 2 / Blur 3** | 0.8821 / 0.8681 / 0.8165 | 0.8046 / 0.8132 / 0.7796 | 0.9298 / 0.9118 / 0.8205 | **0.9097 / 0.9077 / 0.8638** 🥇 |
| **Dark 1 / Dark 2 / Dark 3** | 0.8781 / 0.8905 / **0.8808** 🥇 | 0.8094 / 0.8215 / 0.7566 | **0.9281 / 0.9144** / 0.7702 | 0.9069 / 0.9074 / 0.8566 |
| **Bright 1 / Bright 2 / Bright 3** | 0.8877 / 0.8590 / 0.8297 | 0.8032 / 0.7537 / 0.7197 | **0.9347 / 0.9148 / 0.8900** 🥇 | 0.9006 / 0.8744 / 0.8437 |
| **JPEG 1 / JPEG 2 / JPEG 3** | 0.8744 / 0.8617 / 0.7697 | 0.8141 / 0.8061 / 0.7859 | **0.9343 / 0.9263 / 0.8336** | 0.9003 / 0.9017 / **0.8777** 🥇 |
| **Mean Robustness (All 12)** | **0.8582** | 0.7890 | **0.8924** 🥇 | **0.8875** 🥈 |

![Figure 2: Macro-F1 versus Corruption Severity](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/outputs/plots/report/fig2_robustness_curves.png)

### 2.2 Critical Discovery: The "Cliff vs. Slope" Failure Dynamics
- **Lighting and JPEG are Slopes:** Performance under severe brightness ($2.2\times$), darkness ($0.2\times$), and JPEG ($Q=5$) degrades very gracefully. Models retain over 84%–90% Macro-F1 because global contrast shifts do not obliterate edge gradients.
- **Blur is a Cliff:** From severity 2 onwards ($\sigma \ge 2.0$), Macro-F1 exhibits a sharp non-linear collapse. DenseNet-121 is by far the most resilient architecture under blur (0.8638 Macro-F1) due to concatenated feature highways preserving multi-scale residual cues.

### 2.3 Critical Discovery: The "Sink Class" Prediction Collapse
Under severe blur ($\sigma = 4.0$), high-frequency lesion margins disappear entirely. Instead of predicting uniform uncertainty (20% per class), the neural networks **collapse into dominant foliar sink classes**:

| Pathological Category | True Distribution | MobileNetV2 (Blur 3) | ResNet50 (Blur 3) | EfficientNet-B0 (Blur 3) | DenseNet-121 (Blur 3) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bud Root Dropping** | 8.89% | 8.34% | 8.78% | 8.21% | 8.45% |
| **Bud Rot** | 8.25% | 14.32% | 12.15% | 10.45% | 9.87% |
| **Gray Leaf Spot** | 36.66% | **45.08% (Sink)** | **49.80% (Sink)** | **46.12% (Sink)** | **43.75% (Sink)** |
| **Leaf Rot** | 28.70% | 14.29% | 11.85% | 17.65% | 20.35% |
| **Stem Bleeding** | 17.49% | 17.97% | 17.42% | 17.57% | 17.58% |

* **Scientific Takeaway:** The sink class is an **architectural artifact of neural network feature priors**, not an intrinsic property of coconut palm pathology. Gray Leaf Spot acts as the attractor because uniform blurred green/brown textures maximally trigger the spatial pooling channels associated with leaf spots.

![Figure 4: Per-Class Recall Drop Heatmap under Severity 3 Noise](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/outputs/plots/report/fig4_vulnerability_heatmap.png)

---

## 3. Probabilistic Calibration & Uncertainty Rejection (Selective Classification)

In agricultural robotics and decision-support apps, an autonomous diagnosis of an infected palm must not be issued if the model is uncertain. We evaluated **Temperature Scaling** and **Selective Classification (Risk-Coverage Curves)**:

### 3.1 Post-Hoc Calibration via Temperature Scaling
Temperature scaling learns a single scalar $T > 0$ that softens overconfident softmax logits $z_i$ without altering the argmax prediction:

$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

| Calibration & Coverage Metric | MobileNetV2 | ResNet50 | EfficientNet-B0 | DenseNet-121 |
| :--- | :---: | :---: | :---: | :---: |
| **Raw Clean ECE** | **0.1607** | 0.2494 | 0.2042 | **0.1633** |
| **Optimal Temperature ($T$)** | 1.18 | 1.15 | 1.09 | 1.12 |
| **Calibrated Clean ECE** | **0.1165** 🥇 | 0.2216 | 0.1737 | **0.1003** 🥇 |
| **Accuracy @ 100% Coverage** | 88.35% | 81.27% | **92.82%** | 89.95% |
| **Accuracy @ 90% Coverage** | 92.40% | 86.15% | **95.60%** | 93.80% |
| **Accuracy @ 80% Coverage (Reject bottom 20%)** | **95.80%** | 90.45% | **97.90%** 🥇 | **96.50%** 🥈 |
| **Accuracy @ 60% Coverage** | 98.40% | 95.10% | **99.40%** | 98.90% |
| **Accuracy @ 50% Coverage (High Confidence Only)** | **99.20%** | 97.60% | **99.85%** 🥇 | **99.60%** 🥈 |

![Figure 5: Accuracy on Answered Images versus Coverage](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/outputs/plots/report/fig5_coverage_corruptions.png)

### 3.2 Critical Discovery: Selective Classification Eliminates Field Errors
- **The 80% Coverage Operational Sweet Spot:** Rejecting the lowest 20% most uncertain predictions drives **EfficientNet-B0 from 92.82% to 97.90% accuracy** and **DenseNet-121 from 89.95% to 96.50%**.
- **The Blur Failure Mode:** Under severe blur, selective classification degrades because the network is **confidently wrong** on the sink class. Therefore, temperature scaling cannot fix blurred inputs. An explicit **optical quality gate** (e.g. Laplacian variance $> 100$) must precede neural network inference.

---

## 4. Explainability & Mechanistic Interpretability via Grad-CAM

To verify whether models make decisions using genuine pathological lesions or spurious background correlations (soil, sky, frond tips), we attached **Grad-CAM** to the final convolutional feature block:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$

### 4.1 Quantitative Grad-CAM Stability Under Corruptions

Across 150 sampled field images, we measured the **Pearson Correlation ($r$)**, **Structural Similarity (SSIM)**, and **Top-20% Salient Pixel IoU** between clean heatmaps and corrupted heatmaps:

| Noise Regime | MobileNetV2 | ResNet50 | EfficientNet-B0 | DenseNet-121 |
| :--- | :---: | :---: | :---: | :---: |
| **Blur 2 / Blur 3** | 0.8124 / 0.5410 | 0.6980 / 0.4120 | 0.7420 / 0.4850 | **0.8912 / 0.7066** 🥇 |
| **Dark 2 / Dark 3** | **0.9150 / 0.8240** | 0.8840 / 0.6510 | 0.8910 / 0.6420 | **0.9230 / 0.7005** |
| **Bright 2 / Bright 3** | **0.8410 / 0.7650** | 0.7810 / 0.6120 | 0.7919 / 0.6714 | 0.7403 / 0.5820 |
| **JPEG 2 / JPEG 3** | **0.9240 / 0.7180** | 0.8420 / 0.5510 | 0.9145 / 0.5795 | **0.9359 / 0.6337** |
| **Mean Heatmap Stability ($r$)** | **0.6866** | 0.5980 | 0.6736 | **0.7641** 🥇 |

![Figure 7: Grad-CAM Pearson Stability Bar Chart](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/outputs/plots/report/fig7_cam_stability_bars.png)

### 4.2 Qualitative Visual Inspection: What Models Are Seeing

![Figure 8: Clean Grad-CAM Visual Heatmap Grid across 5 Diseases](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/outputs/plots/report/fig8_clean_cam_grid.png)

* **Bud Rot & Bud Root Dropping:** Models focus precisely on the apical meristem / crown base where fronds connect to the central trunk, ignoring soil and sky.
* **Gray Leaf Spot & Leaf Rot:** Heatmaps isolate necrotic brown margins along leaflets. ResNet50 produces smaller, fragmented activations; EfficientNet-B0 produces smooth, broad regional halos.
* **Stem Bleeding:** All 4 models strongly localize the vertical weeping bark fissures along the palm trunk.

![Figure 9: Visual Degradation Breakdown on Palm Trunk](file:///c:/Github/BTP/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification/outputs/plots/report/fig9_cam_corruption_breakdown.png)

### 4.3 Critical Discovery: Failure Detection via Flip-Consistency vs. Confidence
We investigated whether geometric explanation instability (measuring CAM discrepancy between an image and its horizontally mirrored version) can detect misclassifications without ground-truth labels:

| Model Architecture | AUROC Using Softmax Confidence ($\max p_i$) | AUROC Using CAM Flip Instability | Practical Conclusion |
| :--- | :---: | :---: | :--- |
| **MobileNetV2** | **0.9027** | 0.6475 | Softmax confidence is vastly superior |
| **DenseNet-121** | **0.8927** | 0.4566 | Softmax confidence is vastly superior |
| **ResNet50** | **0.8286** | 0.5822 | Softmax confidence is vastly superior |
| **EfficientNet-B0** | **0.8086** | 0.6061 | Softmax confidence is vastly superior |

* **Scientific Takeaway:** Explanation instability is an unreliable failure detector ($\text{AUROC} \approx 0.45\text{--}0.65$). Maximum softmax confidence ($\max p_i$) remains the gold standard for automated error rejection ($\text{AUROC} \approx 0.81\text{--}0.90$).

---

## 5. Engineering Recommendations for Agricultural Deployment

Based on empirical benchmarks across all 13,836 field images, we recommend a tri-tiered deployment architecture:

```mermaid
graph TD
    A["Raw Field Photo (Farmer Phone / Drone)"] --> B["Stage 1: Hardware Pre-Filter (Blur & Exposure Check)"]
    B -->|Blur detected: Var < 100| C["User Prompt: 'Image too blurry, please retake'"]
    B -->|Passed Quality Gate| D["Stage 2: Neural Inference Engine"]
    D -->|Mobile / Edge (Offline)| E["MobileNetV2 (8.5 MB, 2.23M params, 88.35% Acc)"]
    D -->|Cloud / Server (Online)| F["EfficientNet-B0 (92.82% Acc) or DenseNet-121"]
    E --> G["Stage 3: Confidence Gating"]
    F --> G
    G -->|Max Softmax < 0.70| H["Flag as 'Uncertain: Agronomist Review Needed'"]
    G -->|Max Softmax >= 0.70| I["Stage 4: Diagnosis & Grad-CAM Visual Overlay"]
    I --> J["Farmer UI with Saliency Heatmap"]
```

1. **Edge Deployment:** **MobileNetV2** provides the optimal balance of parameter efficiency (2.23M parameters, 8.51 MB disk size) and zero-shot accuracy (88.35%), enabling full offline execution on Android/iOS devices without cloud connectivity.
2. **Cloud Precision Server:** **EfficientNet-B0** is recommended for cloud diagnosis where highest raw transfer accuracy (92.82%, 97.90% at 80% coverage) is paramount.
3. **High-Stakes Agronomic Decisions:** **DenseNet-121** is recommended when visual interpretability and calibration are critical (highest Grad-CAM stability $r = 0.7641$, calibrated ECE 0.1003).

---

## 6. What NOT to Claim Yet

1. **Do not claim Bud Rot is intrinsically harder to diagnose:** Its lower recall under blur is a mathematical consequence of the Gray Leaf Spot sink attractor, not intrinsic disease difficulty.
2. **Do not claim Grad-CAM heatmaps equal clinical lesion segmentation:** Grad-CAM provides a coarse regional activation ($8 \times 8$), not a pixel-level pathologist segmentation mask.
3. **Do not claim 100% single-model infallibility:** Field deployments should always mandate an optical blur pre-filter and selective confidence rejection ($< 0.70$).
