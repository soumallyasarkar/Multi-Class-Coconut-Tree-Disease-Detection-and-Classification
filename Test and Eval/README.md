# Empirical Robustness, Calibration & Explainability Benchmark Suite
**B.Tech Capstone Project | Department of Computer Science & Engineering | IIIT Kottayam**

This test suite executes the complete empirical framework for evaluating convolutional vision architectures under out-of-distribution corruptions, confidence calibration, Grad-CAM attention stability, and failure detection.

---

## 📁 Module Structure

| Script | Purpose & Functionality |
| :--- | :--- |
| **`corruptions.py`** | Implements the 4 standard corruption families (Gaussian Blur, Darkness, Brightness, JPEG Compression) at 3 severities. Dynamically wraps PyTorch datasets in memory without disk duplication. |
| **`metrics_engine.py`** | Computes Macro-F1 with 95% Bootstrap Confidence Intervals (1,000 resamples), Expected Calibration Error (ECE), Brier Score, Temperature Scaling ($T$), Selective Classification (Risk-Coverage), and AUROC error detection. |
| **`cam_engine.py`** | Hook-based Grad-CAM implementation supporting MobileNetV2, ResNet50, EfficientNet-B0, and DenseNet121. Computes Pearson correlation, SSIM, Top-20% salient overlap (IoU), and horizontal flip-consistency. |
| **`run_experiments.py`** | Master orchestration pipeline that executes clean evaluation, 12-corruption testing, calibration, and Grad-CAM stability on GPU. Dumps structured output to `outputs/experiments/experiment_results.json`. |
| **`generate_plots.py`** | Reads the benchmark JSON and renders publication-ready high-DPI figures (Figures 1 through 9) into `outputs/plots/report/`. |

---

## 🚀 How to Run the Experiments

### 1. Run Benchmark on Default Held-out Test Set
```powershell
python "Test and Eval/run_experiments.py" --device cuda
```

### 2. Run Benchmark on a Completely New / External Dataset
```powershell
python "Test and Eval/run_experiments.py" --data_dir "path/to/new_dataset" --device cuda
```

### 3. Generate Publication Figures for the Report
```powershell
python "Test and Eval/generate_plots.py"
```
Generated figures will appear directly in:
`outputs/plots/report/`
