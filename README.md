# Multi-Class Coconut Tree Disease Detection and Classification

**B.Tech Final Year Project (BTP)**  
*Department of Computer Science and Engineering*  
**Indian Institute of Information Technology Kottayam (IIIT Kottayam)**

---

## 👥 Project Team Members

| Name | Roll Number |
|:---|:---:|
| **Ansh Singh** | `2023BCS0120` |
| **Lakshmi Harshitha Cheedella** | `2023BCS0069` |
| **Soumalya Sarkar** | `2023BCS0172` |
| **Chennupati Bhargav** | `2023BCS0219` |

**Project Guide:** Dr. Lidiya Lilly Thampi

---

## 📑 Table of Contents

- [Introduction](#introduction)
- [Motivation](#motivation)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Dataset Description](#dataset-description)
- [Proposed Methodology](#proposed-methodology)
- [Deep Learning Models](#deep-learning-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Workflow](#project-workflow)
- [Expected Outcomes](#expected-outcomes)
- [Hardware Efficiency & Deployment Feasibility](#hardware-efficiency--deployment-feasibility)
- [Git Branching and Collaboration](#-git-branching-and-collaboration)
- [Future Scope](#future-scope)
- [References](#references)

---

## Introduction

Coconut cultivation plays an important role in agriculture and the livelihoods of farming communities. However, coconut trees are vulnerable to several diseases that affect their leaves, buds, and stems, leading to reduced plant health and productivity.

Identifying these diseases at an early stage is challenging because symptoms vary depending on the disease type, environmental conditions, and stage of infection. Traditional identification methods rely heavily on manual inspection, which can be time-consuming, subjective, and difficult to scale across large plantations.

Deep learning and computer vision offer a promising approach to automating disease recognition through image analysis.

**Multi-Class Coconut Tree Disease Detection and Classification** aims to develop a deep learning framework capable of identifying multiple coconut tree diseases using image-based analysis. The project investigates both image classification and object detection techniques to study disease recognition and the localization of affected regions.

The central focus is a systematic comparison of multiple deep learning architectures under a consistent experimental setup, followed by the selection of a suitable model based on predictive performance and computational efficiency.

## Motivation

Although previous studies have demonstrated promising results in coconut disease classification and detection, important research challenges remain.

- **Limited localization:** Conventional image classification predicts a disease class but does not inherently identify the affected region within an image.
- **Dataset-specific performance:** Models evaluated on different datasets or individual diseases may not generalize equally well to a multi-class setting.
- **Inconsistent comparisons:** Differences in preprocessing, training configurations, and evaluation methods can make model comparisons less reliable.
- **Practical deployment:** High predictive performance must be balanced against inference speed, model size, and computational requirements.

This project addresses these challenges by investigating multiple classification and detection architectures using a common experimental framework and evidence-based model selection.

## Problem Statement

Coconut tree diseases affect different parts of the plant and exhibit varying visual symptoms. Manual identification is often slow and requires domain expertise, making large-scale monitoring difficult.

There is a need to systematically evaluate multiple deep learning models on a five-class coconut disease dataset to determine which architectures are most suitable for disease classification and, where applicable, affected-region detection.

**The objective is to develop and evaluate a deep learning framework for multi-class coconut tree disease classification and detection, compare candidate models fairly, and select the best-performing model using experimental evidence.**

## Project Objectives

1. Prepare and analyze the five-class coconut tree disease dataset.
2. Apply suitable image preprocessing and augmentation techniques.
3. Implement and train multiple deep learning classification and detection architectures.
4. Evaluate candidate models under a consistent experimental setup wherever applicable.
5. Compare models using accuracy, precision, recall, F1-score, mAP, and other task-specific metrics.
6. Analyze class-wise performance, misclassifications, and detection errors.
7. Examine model size, computational cost, and inference efficiency.
8. Select the most suitable model based on comparative evaluation.
9. Develop and evaluate the final disease classification and/or detection pipeline.

---

## Dataset Description

The project uses the **Coconut Tree Disease Dataset** published by Thite et al. (2023).

The dataset contains **5,798 images across five coconut disease classes**.

### Dataset Distribution

| Disease Class | Number of Images | Percentage |
|:---|---:|---:|
| Bud Root Dropping | 514 | 8.86% |
| Bud Rot | 470 | 8.11% |
| Gray Leaf Spot | 2,135 | 36.82% |
| Leaf Rot | 1,673 | 28.85% |
| Stem Bleeding | 1,006 | 17.35% |
| **Total** | **5,798** | **100%** |

*Percentages are rounded to two decimal places.*

### Dataset Characteristics

- **Total images:** 5,798
- **Number of classes:** 5
- **Task:** Multi-class disease classification and investigation of object detection approaches
- **Primary challenge:** Differences in sample counts across disease classes

### Dataset Preparation

The dataset preparation process will consider:

- Image quality and consistency.
- Appropriate training, validation, and test splits.
- Image resizing and normalization.
- Suitable augmentation strategies.
- Class-wise sample distribution.
- Annotation requirements for object detection models.

The same underlying dataset and comparable data-splitting procedures will be used where appropriate to support meaningful comparisons between candidate models.

**Dataset Reference:** Thite, S., Suryawanshi, Y., Patil, K., & Chumchu, P. (2023). *Coconut tree disease dataset*. Data in Brief, 51, 109690.

[Read the dataset publication](https://doi.org/10.1016/j.dib.2023.109690)

---

## Proposed Methodology

The proposed system follows a structured deep learning pipeline, from dataset preparation to final model selection.

### 1. Dataset Preparation and Analysis

Inspect the dataset, examine class distributions, identify potential data-quality issues, and prepare appropriate data splits.

### 2. Image Preprocessing and Augmentation

Apply suitable resizing, normalization, enhancement, and augmentation techniques. Prepare disease-region annotations where required for object detection.

### 3. Model Implementation and Training

Implement candidate CNN-based classification architectures and object detection models. Train and validate each candidate using appropriate configurations.

### 4. Comparative Evaluation

Evaluate classification and detection models using task-specific metrics. Examine class-wise results, prediction errors, localization quality, and computational requirements.

### 5. Best Model Selection

Compare experimental results and select the model that best satisfies the project's accuracy, reliability, and efficiency requirements.

### 6. Final System Development

Integrate the selected model into a final disease classification and/or detection pipeline capable of processing coconut tree images and producing the corresponding predictions.

---

## Deep Learning Models

The project investigates multiple established deep learning architectures for comparative evaluation.

### Image Classification Models

| Architecture | Overview |
|:---|:---|
| **VGG16** | CNN architecture using sequential convolutional layers. |
| **ResNet50** | Deep residual network designed to improve optimization through residual connections. |
| **MobileNetV2** | Lightweight CNN architecture designed for computational efficiency. |
| **EfficientNet** | Architecture family that balances network depth, width, and input resolution. |
| **DenseNet** | CNN architecture with dense connections between feature-processing layers. |

### Object Detection Models

| Architecture | Overview |
|:---|:---|
| **YOLOv5** | Object detection architecture designed for efficient, single-stage detection. |
| **YOLOv8** | YOLO-family architecture supporting modern object detection workflows. |
| **Faster R-CNN** | Region-proposal-based object detection architecture for localization and classification. |

### Comparative Study

The candidate models will be evaluated according to their respective task requirements.

Classification models predict the disease class, while object detection models can additionally produce bounding boxes around detected regions when trained with suitable annotations.

The comparison will focus on:

- Predictive performance.
- Class-wise recognition capability.
- Localization quality for detection models.
- Model complexity and computational cost.
- Suitability for practical deployment.

**The final model will be selected based on observed experimental results rather than an assumed ranking of architectures.**

---

## Evaluation Metrics

A combination of predictive-performance and computational-efficiency metrics will be used to assess the candidate models.

### Classification Metrics

| Metric | Purpose |
|:---|:---|
| **Accuracy** | Measures the proportion of correctly classified samples. |
| **Precision** | Measures the proportion of predicted positive samples that are correct. |
| **Recall** | Measures the proportion of actual positive samples correctly identified. |
| **F1-score** | Balances precision and recall using their harmonic mean. |
| **Confusion Matrix** | Shows correct predictions and misclassifications across classes. |

For multi-class classification, precision, recall, and F1-score will be calculated using an appropriate averaging strategy.

### Object Detection Metrics

| Metric | Purpose |
|:---|:---|
| **Mean Average Precision (mAP)** | Summarizes detection performance across classes under specified evaluation thresholds. |
| **Bounding-Box Localization** | Measures how accurately the detected regions align with the target regions. |

### Computational Efficiency Metrics

- **Model Size:** Storage requirements for the trained model.
- **Inference Speed:** Time or frames processed per second under specified conditions.
- **Computational Cost:** Resources required for training and inference.
- **Deployment Feasibility:** Suitability for systems with limited computational resources.

Classification and detection metrics will be interpreted separately where necessary, since the two tasks produce different types of predictions.

---

## Project Workflow

The planned project follows these phases.

| Phase | Activity |
|:---:|:---|
| **1** | Dataset collection, cleaning, and exploratory analysis |
| **2** | Data preprocessing, augmentation, and annotation preparation |
| **3** | Implementation of candidate classification and detection models |
| **4** | Model training, validation, and hyperparameter tuning |
| **5** | Comparative experiments and class-wise performance analysis |
| **6** | Selection of the best-performing model and development of the final system |
| **7** | Performance analysis, error analysis, and final evaluation |

This workflow is intended to maintain a clear progression from data preparation through model evaluation and final system development.

---

## Expected Outcomes

The project aims to deliver:

- A prepared and documented five-class coconut disease dataset.
- Multiple trained deep learning classification and detection models.
- A systematic comparison of candidate architectures.
- Quantitative evaluation reports using appropriate performance metrics.
- Class-wise performance analysis and confusion matrices.
- Analysis of computational efficiency and model complexity.
- An evidence-based selection of the final model.
- A final disease classification and/or detection pipeline.

These are intended deliverables. Final performance figures and conclusions will be reported after the corresponding experiments have been conducted.

---

## Hardware Efficiency & Deployment Feasibility

Practical agricultural applications require models that balance predictive performance with computational efficiency.

Alongside classification and detection performance, the project considers the feasibility of deploying the selected model in resource-constrained environments.

Key considerations include:

- Model size and storage requirements.
- Inference latency and throughput.
- Hardware requirements.
- Optimization opportunities for faster inference.
- Trade-offs between model complexity and predictive performance.

Where edge inference is evaluated, the reported FPS, model size, hardware configuration, input resolution, and inference conditions should be documented to ensure that the results are reproducible and meaningful.

---

## 🌿 Git Branching and Collaboration

The repository follows a branch-based collaboration workflow. Each team member develops and maintains their work on an individual branch, while `main` serves as the shared branch for general project documentation and reviewed integration.

### Branch Structure

| Branch | Team Member | Purpose |
|:---|:---|:---|
| `main` | Shared project branch | General documentation and approved integration |
| `ansh` | Ansh Singh | Individual development and experiments |
| `lakshmi` | Lakshmi Harshitha Cheedella | Individual development and experiments |
| `soumallya` | Soumalya Sarkar | Individual development and experiments |
| `bhargav` | Chennupati Bhargav | Individual development and experiments |

*The branch names above are the agreed naming convention for the project. Git branch names are case-sensitive, so members should use the exact names adopted by the team.*

### Clone the Repository

```bash
git clone https://github.com/soumallyasarkar/Multi-Class-Coconut-Tree-Disease-Detection-and-Classification.git

cd Multi-Class-Coconut-Tree-Disease-Detection-and-Classification
```

### Switch to Your Branch

For example, to work on the `soumallya` branch:

```bash
git switch soumallya
```

Replace `soumallya` with your assigned branch name.

If your branch exists on GitHub but not locally, create a local tracking branch:

```bash
git fetch origin
git switch --track origin/soumallya
```

### Get the Latest Changes

```bash
git pull origin soumallya
```

Use your own branch name in place of `soumallya`.

### Commit and Push Your Work

```bash
git add .
git commit -m "Describe your changes"
git push origin soumallya
```

Replace the branch name with your assigned branch.

### Contribution Guidelines

1. **Work independently:** Make code and experiment changes on your assigned branch.
2. **Protect `main`:** Avoid pushing unreviewed experimental work directly to the shared branch.
3. **Use meaningful commits:** Write commit messages that describe the changes made.
4. **Keep experiments reproducible:** Document the relevant configurations, datasets, and evaluation procedures.
5. **Review before integration:** Test completed work and submit a pull request when it is ready to merge.
6. **Resolve conflicts carefully:** Coordinate changes that affect shared files or model interfaces.
7. **Manage large files appropriately:** Avoid committing datasets, checkpoints, and generated artifacts unnecessarily.

This workflow allows team members to develop independently while keeping the shared project history organized and maintainable.

---

## Future Scope

The project may be extended in several directions:

- **Healthy Class Integration:** Introduce a healthy coconut tree class to distinguish healthy plants from diseased plants.
- **Real-World Validation:** Evaluate the final models on new field images captured under practical agricultural conditions.
- **Lightweight Architectures:** Investigate efficient architectures and model optimization techniques for edge deployment.
- **Disease Severity Estimation:** Explore methods for estimating the severity of disease symptoms.
- **Robustness Analysis:** Study model performance under different lighting conditions, backgrounds, image qualities, and disease stages.
- **Practical Decision Support:** Explore ways to present predictions and localized disease regions in a useful format for farmers and researchers.

---

## References

1. Thite, S., Suryawanshi, Y., Patil, K., & Chumchu, P. (2023). *Coconut tree disease dataset*. Data in Brief, 51, 109690. [https://doi.org/10.1016/j.dib.2023.109690](https://doi.org/10.1016/j.dib.2023.109690)

2. Hiremath, J. S., et al. (2025). *Advanced Detection of Coconut Stem Bleeding Using an Improved YOLOv5 Model for Precision Agriculture*. IEEE INCET 2025. [https://doi.org/10.1109/INCET64471.2025.11139861](https://doi.org/10.1109/INCET64471.2025.11139861)

3. Subbaian, S., et al. (2024). *Detection of coconut leaf diseases using enhanced deep learning techniques*. Journal of Intelligent & Fuzzy Systems, 46(2), 5033–5045.

4. Singh, P., Verma, A., & Alex, J. S. R. (2021). *Disease and pest infection detection in coconut tree through deep learning techniques*. Computers and Electronics in Agriculture, 182, 105986.

5. Vidhanaarachchi, S., et al. (2025). *Early Diagnosis and Severity Assessment of Weligama Coconut Leaf Wilt Disease and Coconut Caterpillar Infestation Using Deep Learning-Based Image Processing Techniques*. IEEE Access, 13.

6. Nesarajan, D., et al. (2020). *Coconut Disease Prediction System Using Image Processing and Deep Learning Techniques*. IEEE IPAS 2020.

7. Murthy, G. N. K., et al. (2024). *Early Detection of Stem Bleeding in Coconut Trees using Image Processing*. IEEE ICRASET 2024.

8. Kavithamani, V., & UmaMaheswari, S. (2023). *Investigation of deep learning for whitefly identification in coconut tree leaves*. Intelligent Systems with Applications, 20, 200290.

9. Megalingam, R. K., et al. (2022). *Analysis of Coconut Tree Trunk for the Identification of Cracks and Holes Using Image Processing*. IEEE I2CT 2022.

10. Bochkovskiy, A., Wang, C.-Y., & Liao, H.-Y. (2020). *YOLOv4: Optimal Speed and Accuracy of Object Detection*. arXiv:2004.10934. [https://arxiv.org/abs/2004.10934](https://arxiv.org/abs/2004.10934)

---

## Acknowledgements

We express our gratitude to **Dr. Lidiya Lilly Thampi**, our project guide, for her guidance and support throughout this work.

We also acknowledge the **Indian Institute of Information Technology Kottayam** for providing the academic environment and opportunity to undertake this project.

---

<div align="center">

**Multi-Class Coconut Tree Disease Detection and Classification**

*B.Tech Final Year Project — Department of Computer Science and Engineering*

**Indian Institute of Information Technology Kottayam**

</div>
