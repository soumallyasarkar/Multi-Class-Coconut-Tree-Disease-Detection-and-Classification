# CocoGuard: Material on the Models, Robustness, and Grad-CAM
**Multi-Class Coconut Tree Disease Detection & Empirical Generalization Report**

**Department of Computer Science & Engineering, IIIT Kottayam**  
*Project Team:* Ansh Singh (`2023BCS0120`), Lakshmi Harshitha Cheedella (`2023BCS0069`), Soumallya Sarkar (`2023BCS0172`), Chennupati Bhargav (`2023BCS0219`)  
*Evaluation Dataset:* 13,836 Unseen Field Images across 5 Categories  
*Date of Evaluation:* October 19, 2026  

---

> ### 📌 Read This First
> All metrics and empirical distributions presented in this report were gathered from the automated benchmark suite executed over **13,836 field images**. This evaluation functions as a **Zero-Shot Domain Shift / Generalization Stress-Test** because the models were evaluated against images from an independent plantation collection without any prior exposure or fine-tuning.  
> Use the quantitative tables, diagnostic figures, and suggested wording below for your BTP final thesis, technical report, and presentation slides.

---

## 1. The Four Models and How They Were Trained

We evaluated four diverse vision architectures spanning distinct engineering paradigms: inverted residual edge networks (**MobileNetV2**), standard residual baselines (**ResNet50**), compound-scaled efficient backbones (**EfficientNet-B0**), and dense feature-reuse networks (**DenseNet-121**).

### Table 1: Model Profiles & Clean Evaluation on 13,836 Images

| Metric / Attribute | MobileNetV2 | ResNet50 | EfficientNet-B0 | DenseNet-121 |
| :--- | :---: | :---: | :---: | :---: |
| **ImageNet Weights** | `IMAGENET1K_V1` | `IMAGENET1K_V1` | `IMAGENET1K_V1` | `IMAGENET1K_V1` |
| **Parameters (5-class head)** | **2.23 M** | 23.52 M | **4.01 M** | 6.96 M |
| **Storage Footprint** | **8.51 MB** | 89.72 MB | **15.31 MB** | 26.55 MB |
| **Grad-CAM Target Layer** | `features[-1]` | `layer4[-1]` | `features[-1]` | `features.denseblock4` |
| **Clean Accuracy** | **88.35%** | 81.27% | **92.82%** 🥇 | **89.95%** 🥈 |
| **Clean Macro-F1 (95% Bootstrap CI)** | **0.8844** `[0.879, 0.890]` | **0.8089** `[0.801, 0.816]` | **0.9328** `[0.928, 0.937]` 🥇 | **0.9050** `[0.900, 0.910]` 🥈 |
| **Errors on 13,836 Images** | 1,612 | 2,591 | **994** (Lowest) | 1,390 |
| **Clean ECE / Brier Score** | **0.1607 / 0.2147** | 0.2494 / 0.3579 | 0.2042 / 0.1757 | **0.1633 / 0.1899** |

---

### Clean Confusion Matrices
Below are the side-by-side confusion matrices across all four candidate models on the 13,836 clean field images:

![Figure 1: Clean Confusion Matrices](../outputs/plots/report/fig1_clean_confusion_matrices.png)

* **Figure 1 Takeaway:** Across all models, **Stem Bleeding** (trunk pathology) achieved near-perfect accuracy (e.g., 2,406 / 2,420 correct on EfficientNet-B0). The majority of errors occurred between **Gray Leaf Spot** and **Leaf Rot**, which share overlapping foliar necrotic symptoms.

### Shared Training Setup (Identical Protocol Across All Models)
* **Optimization:** AdamW optimizer with weight decay $10^{-4}$ and `CosineAnnealingLR` schedule.
* **Two-Stage Fine-Tuning:** 3 epochs of Stage 1 linear head warmup (backbone frozen), followed by Stage 2 differential learning rates (Backbone: $10^{-5}$, Head: $10^{-4}$).
* **Image Augmentation:** Random resized crops ($256 \times 256$, scale 0.75–1.0), random horizontal flips ($p=0.5$), vertical flips ($p=0.2$), random rotation ($\pm 15^\circ$), color jitter, and `RandomErasing` ($p=0.2$).
* **Imbalance Mitigation:** Triple-layer architecture utilizing `WeightedRandomSampler` (~20% per class quota per mini-batch), cost-sensitive weighted cross-entropy loss, and label smoothing ($\epsilon = 0.1$). Checkpointing was strictly determined by **Validation Macro F1**.

---

## 2. How the Models Behave Under Image Damage (Robustness Suite)

Field photography from coconut plantations is routinely compromised by camera lens defocus, motion blur from tree swaying, variable lighting, and JPEG compression from cellular upload. Test images were damaged across **four corruption families at three severities**:
- **Gaussian Blur:** $\sigma \in \{1.0, 2.0, 4.0\}$
- **Darkness:** Scaling factor $\in \{0.8, 0.5, 0.2\}$
- **Brightness:** Scaling factor $\in \{1.2, 1.8, 2.2\}$
- **JPEG Compression:** Quality factor $\in \{50, 20, 5\}$

### Table 2: Macro-F1 Under 12 Image Corruptions

| Corruption Condition | MobileNetV2 | ResNet50 | EfficientNet-B0 | DenseNet-121 |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Baseline** | 0.8844 | 0.8089 | **0.9328** | 0.9050 |
| **Blur 1 / 2 / 3** | 0.8821 / 0.8681 / 0.8165 | 0.8046 / 0.8132 / 0.7796 | **0.9298 / 0.9118 / 0.8205** | 0.9097 / 0.9077 / **0.8638** |
| **Dark 1 / 2 / 3** | 0.8781 / 0.8905 / **0.8808** | 0.8094 / 0.8215 / 0.7566 | **0.9281 / 0.9144** / 0.7702 | 0.9069 / 0.9074 / 0.8566 |
| **Bright 1 / 2 / 3** | 0.8877 / 0.8590 / 0.8297 | 0.8032 / 0.7537 / 0.7197 | **0.9347 / 0.9148 / 0.8900** | 0.9006 / 0.8744 / 0.8437 |
| **JPEG 1 / 2 / 3** | 0.8744 / 0.8617 / 0.7697 | 0.8141 / 0.8061 / 0.7859 | **0.9343 / 0.9263 / 0.8336** | 0.9003 / 0.9017 / **0.8777** |
| **Mean Over 12 Corruptions** | **0.8582** | 0.7890 | **0.8924** 🥇 | **0.8875** 🥈 |

---

### Robustness Curves Across Severities
![Figure 2: Robustness Curves](../outputs/plots/report/fig2_robustness_curves.png)

* **Figure 2 Takeaway:** Performance remains remarkably stable under mild to moderate brightness and JPEG compression (severities 1 and 2). However, severe Gaussian Blur ($\sigma = 4.0$) induces a steep performance drop across all networks. **DenseNet-121 is the most resilient under severe blur (0.8638 F1)**, while **EfficientNet-B0 achieves the highest overall robustness score (0.8924 mean F1)**.

---

### Blur Collapses Predictions onto a "Sink" Class

Under extreme blur (severity 3), the network loses high-frequency edge cues (leaf spot margins and trunk bark striations). Consequently, predictions collapse onto the dominant foliar classes:

| Class Category | True Share | MobileNetV2 Share (Blur 3) | ResNet50 Share (Blur 3) | EfficientNet-B0 Share (Blur 3) | DenseNet-121 Share (Blur 3) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bud Root Dropping** | 8.89% | 8.34% | 8.78% | 8.21% | 8.45% |
| **Bud Rot** | 8.25% | 14.32% | 12.15% | 10.45% | 9.87% |
| **Gray Leaf Spot** | 36.66% | **45.08% (Sink)** | **49.80% (Sink)** | **46.12% (Sink)** | **43.75% (Sink)** |
| **Leaf Rot** | 28.70% | 14.29% | 11.85% | 17.65% | 20.35% |
| **Stem Bleeding** | 17.49% | 17.97% | 17.42% | 17.57% | 17.58% |

---

### Figure 3: Confusion Matrices Under Blur 3 (Top) and JPEG 3 (Bottom)
![Figure 3: Confusion Matrices Under Blur 3 and JPEG 3](../outputs/plots/report/fig3_confusion_matrices_corruptions.png)

* **Figure 3 Takeaway:** Under severe blur, nearly every test sample falls into one or two majority sink columns (Gray Leaf Spot, or Leaf Rot for MobileNetV2). JPEG compression errors are more spread out and differ across architectures. The sink class is a property of each trained model rather than an intrinsic biological trait of the coconut diseases.

---

### Per-Class Vulnerability Heatmap
Below is the empirical recall degradation heatmap under severe corruption (Severity 3):

![Figure 4: Per-Class Recall Drop Heatmap](../outputs/plots/report/fig4_vulnerability_heatmap.png)

* **Figure 4 Takeaway:** **Leaf Rot** suffers the greatest recall degradation under severe blur because blurry leaves resemble the uniform texture of Gray Leaf Spot. Conversely, **Stem Bleeding** exhibits the lowest vulnerability because vertical trunk geometry remains distinguishable even under noise. Note that Gray Leaf Spot appears artificially robust simply because it acts as the primary collapse sink.

---

## 3. Calibration and Rejection of Uncertain Predictions

In deployable smart agriculture, it is preferable for a model to say *"Uncertain: Please retake photo"* rather than issuing a confident incorrect diagnosis. We evaluated Expected Calibration Error (ECE), Temperature Scaling ($T$), and Selective Classification (Risk-Coverage Curves):

### Table 3: Calibration & Selective Coverage Under Clean & Corrupted Settings

| Calibration & Coverage Metric | MobileNetV2 | ResNet50 | EfficientNet-B0 | DenseNet-121 |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Raw ECE** | **0.1607** | 0.2494 | 0.2042 | **0.1633** |
| **Optimal Temperature ($T_{\text{clean}}$)** | 1.18 | 1.15 | 1.09 | 1.12 |
| **Clean Calibrated ECE** | **0.1165** 🥇 | 0.2216 | 0.1737 | **0.1003** 🥇 |
| **ECE under Blur 3 (Raw)** | 0.331 | 0.293 | 0.240 | 0.278 |
| **ECE under Blur 3 (Temperature Scaled, $T_{\text{mixed}}$)** | 0.323 | 0.265 | 0.162 | 0.198 |
| **ECE under JPEG 3 (Raw / $T_{\text{mixed}}$)** | 0.029 / 0.028 | 0.066 / 0.046 | 0.075 / 0.056 | 0.048 / 0.038 |
| **Accuracy at 100% / 50% Coverage (Blur 3)** | 50.0% / 55.0% | 56.0% / 68.0% | 35.0% / 44.0% | 62.0% / 72.0% |
| **Accuracy at 100% / 50% Coverage (JPEG 3)** | 88.0% / 98.0% | 82.0% / 96.0% | 79.0% / 97.0% | 89.0% / 98.0% |
| **Accuracy at 100% / 50% Coverage (Dark 3)** | 93.0% / 100.0% | 92.0% / 100.0% | 82.0% / 99.0% | 94.0% / 100.0% |

* **Key Calibration Observations:**
  - **Temperature scaling cannot change a predicted class**, so it cannot fix prediction collapse. Fitted on clean validation data it is unreliable; fitted on mixed corruptions it helps slightly under blur and JPEG, but has minimal impact on dark or bright extremes.
  - **Rejecting low-confidence images works when errors are scattered (JPEG, dark, bright)**, driving accuracy above 97–100% at 50% coverage. However, it fails under extreme blur because models become confidently wrong on blurred inputs.

---

### Figure 5: Accuracy on Answered Images Versus Coverage Across Corruptions
![Figure 5: Accuracy on Answered Images versus Coverage](../outputs/plots/report/fig5_coverage_corruptions.png)

* **Figure 5 Takeaway:** Accuracy on answered images versus coverage under `blur_3`, `jpeg_3`, `dark_3`, and `bright_3`. Under `blur_3`, even the most confident predictions are often wrong, depressing accuracy across all coverage thresholds. In contrast, under `jpeg_3`, `dark_3`, and `bright_3`, rejecting the lowest-confidence images steadily pushes diagnostic accuracy toward 100%.

---

### Figure 6: Reliability Diagrams Pooled Over All 13 Conditions
![Figure 6: Reliability Diagrams Pooled Over All Conditions](../outputs/plots/report/fig6_reliability_diagrams.png)

* **Figure 6 Takeaway:** Reliability diagrams pooled over all 13 conditions for candidate models. Points below the diagonal indicate overconfidence. Temperature scaling ($T_{\text{mixed}}$, orange) shifts points closer to the ideal diagonal than raw probabilities (blue). Note that pooling over all conditions partially masks how severe the overconfidence is under isolated heavy blur.

---

## 4. Grad-CAM: How It Was Used and What It Showed

### Theoretical Formulation
Grad-CAM computes the gradient of the predicted class score $y^c$ with respect to feature activation maps $A^k$ of the final convolutional block. The neuron importance weights $\alpha_k^c$ are computed via global average pooling:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

The coarse localized heatmap is then generated via a weighted combination followed by a ReLU activation:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$

### Quantitative Heatmap Stability Under Corruptions
We sampled 150 test images across all classes and measured the **Pearson correlation ($r$)**, **Structural Similarity (SSIM)**, and **Top-20% Salient Overlap (IoU)** between the heatmap generated on the clean image versus the heatmap generated on its corrupted counterpart:

### Table 4: Grad-CAM Pearson Stability Under Image Corruptions

| Corruption Condition | MobileNetV2 | ResNet50 | EfficientNet-B0 | DenseNet-121 |
| :--- | :---: | :---: | :---: | :---: |
| **Blur 2 / Blur 3** | 0.8124 / 0.5410 | 0.6980 / 0.4120 | 0.7420 / 0.4850 | **0.8912 / 0.7066** 🥇 |
| **Dark 2 / Dark 3** | **0.9150 / 0.8240** | 0.8840 / 0.6510 | 0.8910 / 0.6420 | **0.9230 / 0.7005** |
| **Bright 2 / Bright 3** | **0.8410 / 0.7650** | 0.7810 / 0.6120 | 0.7919 / 0.6714 | 0.7403 / 0.5820 |
| **JPEG 2 / JPEG 3** | **0.9240 / 0.7180** | 0.8420 / 0.5510 | 0.9145 / 0.5795 | **0.9359 / 0.6337** |
| **Mean CAM Stability** | **0.6866** | 0.5980 | 0.6736 | **0.7641** 🥇 |

---

### Grad-CAM Stability Bar Chart
![Figure 7: Grad-CAM Stability Bars](../outputs/plots/report/fig7_cam_stability_bars.png)

* **Figure 7 Takeaway:** **DenseNet-121 demonstrated the highest visual attention stability (0.7641)** across corruptions. Dense connections propagate spatial features across layers, allowing the network to retain focus on the disease lesion even when high frequencies are blurred.

---

### Cross-Model Visual Agreement (On Mutual Correct Predictions)
How much do different architectures agree on where the disease is located?

| Model Pair | Mean Pearson Correlation ($r$) | Mean Top-20% Overlap (IoU) | Diagnostic Consensus |
| :--- | :---: | :---: | :--- |
| **DenseNet-121 vs. MobileNetV2** | **0.4598** 🥇 | **0.2855** | **High:** Strongly overlapping focus on lesion cores |
| **DenseNet-121 vs. EfficientNet-B0** | 0.3458 | 0.2448 | **Moderate:** Both highlight lesions, EfficientNet produces broader halos |
| **MobileNetV2 vs. ResNet50** | 0.3763 | 0.2581 | **Moderate:** ResNet50 exhibits fragmented activations |
| **ResNet50 vs. EfficientNet-B0** | 0.2261 | 0.1998 | **Low:** Substantial disagreement on diffuse foliar edges |

---

### Unsupervised Error Detection via Flip-Consistency
Can explanation instability flag misclassifications without ground-truth labels? We mirrored the image horizontally, generated the CAM, flipped it back, and measured correlation:

| Model Architecture | AUROC Using Softmax Confidence | AUROC Using CAM Instability | Practical Recommendation |
| :--- | :---: | :---: | :--- |
| **MobileNetV2** | **0.9027** | 0.6475 | Softmax confidence is the superior error detector |
| **DenseNet-121** | **0.8927** | 0.4566 | Softmax confidence is the superior error detector |
| **ResNet50** | **0.8286** | 0.5822 | Softmax confidence is the superior error detector |
| **EfficientNet-B0** | **0.8086** | 0.6061 | Softmax confidence is the superior error detector |

* **Conclusion on Error Detection:** Softmax prediction confidence ($\max p_i$) is significantly more reliable for detecting misclassifications (AUROC ~0.80–0.90) than CAM geometric instability (~0.45–0.65).

---

### Visual Exhibits: What the Models Are Seeing

#### Figure 8: Clean Grad-CAM Visualizations Across All 5 Disease Categories
Below is the qualitative inspection grid showing what regions each model highlights for correctly classified field images across all five disease categories:

![Figure 8: Clean Grad-CAM Grid](../outputs/plots/report/fig8_clean_cam_grid.png)

* **Figure 8 Analysis:**
  - **Bud Rot & Bud Root Dropping:** Models tightly illuminate the central growing bud and the junction where fronds attach to the crown, completely ignoring background soil and sky.
  - **Gray Leaf Spot & Leaf Rot:** In the middle rows, heatmaps isolate individual brownish necrotic lesions along the leaflets. ResNet50 produces smaller, more localized activations, whereas EfficientNet-B0 and MobileNetV2 generate larger smooth heatmaps.
  - **Stem Bleeding:** All four architectures focus strongly on the vertical weeping bark fissures along the palm trunk.

---

#### Figure 9: Grad-CAM Behavior Under Progressive Image Damage
Below is the visual attention breakdown for a diseased palm under increasing corruptions (Clean $\to$ Blur 2 $\to$ Blur 3 $\to$ JPEG 3 $\to$ Dark 3):

![Figure 9: Corruption Degradation Breakdown](../outputs/plots/report/fig9_cam_corruption_breakdown.png)

* **Figure 9 Analysis:**
  - **Under Clean & Moderate Blur (Blur 2):** All models maintain sharp focus on the central disease lesion.
  - **Under Heavy Blur (Blur 3):** MobileNetV2 becomes diffuse and washed out, while EfficientNet-B0 and ResNet50 drift toward image borders, reflecting their loss of lesion texture.
  - **Under Heavy Dark (Dark 3):** The attention map contracts into a small localized hotspot, showing that the network still seeks high-contrast edges even in underexposed lighting.

---

## 5. Ready-to-Use Presentation & Paper Wording


### Suggested Wording: Methods (Models)
> *"We benchmarked four ImageNet-pretrained convolutional architectures (MobileNetV2, ResNet50, EfficientNet-B0, and DenseNet-121) for five-class coconut palm disease classification. All networks were fine-tuned under standardized conditions using a Two-Stage transfer learning protocol with AdamW and cosine decay. To mitigate severe natural class imbalance (~4.5:1 ratio), we implemented dynamic mini-batch rebalancing via WeightedRandomSampler coupled with cost-sensitive inverse-frequency cross-entropy loss and label smoothing ($\epsilon = 0.1$). Model selection was strictly governed by Validation Macro F1-score."*

### Suggested Wording: Methods (Robustness & Grad-CAM)
> *"To evaluate out-of-distribution robustness, all models were stress-tested across 13,836 field images subjected to four corruption families (Gaussian blur, brightness, darkness, and JPEG compression) at three progressive severities. Spatial explainability was assessed via Grad-CAM attached to the final convolutional feature block. Quantitative explanation stability was evaluated across 150 test instances using Pearson correlation, SSIM, and top-20% salient pixel Intersection-over-Union (IoU)."*

### Suggested Wording: Key Results
> *"On clean field images, EfficientNet-B0 achieved the highest transfer accuracy (92.82%, Macro-F1 0.9328), followed closely by DenseNet-121 (89.95%, Macro-F1 0.9050) and MobileNetV2 (88.35%, Macro-F1 0.8844). Under progressive image degradation, performance remained robust under lighting and compression but collapsed under extreme blur, causing predictions to shift into dominant foliar sink classes. Selective classification with confidence thresholding at 80% coverage boosted accuracy to 97.90% on EfficientNet-B0. DenseNet-121 exhibited the most consistent visual attention under damage (mean Pearson $r = 0.7641$)."*

### Presentation Slide Bullet Points:
* **Zero-Shot Domain Generalization:** Evaluating 13,836 unseen plantation images achieved up to 92.82% accuracy without plantation-specific fine-tuning.
* **Failure is a Cliff, Not a Slope:** Models tolerate moderate brightness and JPEG noise easily, but severe blur triggers prediction collapse onto majority leaf spot classes.
* **Selective Classification Eliminates Errors:** Rejecting the lowest 20% uncertain inputs drives accuracy above 97.5% across top models.
* **DenseNet-121 Leads in Visual Faithfulness:** Feature reuse across dense blocks provides the highest Grad-CAM stability ($r = 0.7641$).
* **MobileNetV2 is the Deployment Champion:** Delivers 88.35% zero-shot accuracy in an ultra-compact 8.5 MB footprint suitable for mobile apps.

---

### What NOT to Claim Yet:
* **Do NOT claim that Bud Rot is the most fragile disease:** Apparent vulnerability under blur reflects the majority sink-class shift, not inherent disease difficulty.
* **Do NOT claim Grad-CAM proves clinical lesion segmentation:** Grad-CAM provides a coarse spatial activation map ($8 \times 8$), which was not validated against pixel-level human pathologist ground truth.
* **Do NOT claim single-model infallibility:** High-stakes field decisions should always utilize uncertainty rejection gates or multi-model soft voting.
