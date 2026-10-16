# Next Steps — Coconut Tree Disease Detection Project

**Last Updated:** October 13, 2026  
**Where we are now:** All 4 phases of code are written and validated. The isolated `.venv` is set up. A mock pipeline test was run successfully (`mobilenet_v2` checkpoint exists).  
**What comes next:** Load real data → Train all models → Evaluate → Explain

---

## Step 1: Load the Real Dataset

> [!IMPORTANT]
> Currently the project is running on **synthetic/mock images** (15 per class). The model checkpoint in `outputs/checkpoints/` is meaningless until you train on real data.

### 1.1 — Get the Dataset
Download the **Coconut Tree Disease dataset** from Thite et al. (2023) (or your dataset source). You need **5,798 images** organized in the following structure:

```
data/
└── raw/
    ├── Bud Root Dropping/   (≈514 images)
    ├── Bud Rot/             (≈470 images)
    ├── Gray Leaf Spot/      (≈2135 images)
    ├── Leaf Rot/            (≈1673 images)
    └── Stem Bleeding/       (≈1006 images)
```

> [!TIP]
> Folder names must match **exactly** (case-sensitive). The code also handles `lowercase` and `underscore` variants automatically.

### 1.2 — Verify & Split the Dataset

Once images are in place, run the audit + split command:

```powershell
.\.venv\Scripts\Activate.ps1
python prepare_data.py --split
```

> [!TIP]
> **Balanced Undersampling Mode (Default: 470 images per class)**:
> Because the minority label (*Bud Rot*) contains 470 images while majority labels have up to 2,135, `config.SAMPLES_PER_CLASS` is configured to `470`. Each class is capped at 470 images, producing a balanced dataset of 2,350 total images.
> To use the full natural imbalanced dataset instead, pass `--samples_per_class 0`.

**Expected output (Balanced Mode — 470 images/label):**
```
Auditing COCONUT DISEASE DATASET...
Discovered 5798 total images:
  - Bud Root Dropping   :   514 images (  8.9%) | [Paper benchmark: 514] -> Balanced to 470
  - Bud Rot             :   470 images (  8.1%) | [Paper benchmark: 470] -> Balanced to 470
  - Gray Leaf Spot      :  2135 images ( 36.8%) | [Paper benchmark: 2135] -> Balanced to 470
  - Leaf Rot            :  1673 images ( 28.9%) | [Paper benchmark: 1673] -> Balanced to 470
  - Stem Bleeding       :  1006 images ( 17.3%) | [Paper benchmark: 1006] -> Balanced to 470

[BALANCED MODE] Balancing active: capping each class at 470 images (matching minority class).

Performing Stratified 70% Train / 15% Val / 15% Test Split...
Stratified Splits Saved to data\splits:
  Total images used: 2350 (out of 5798 available raw images)
  Train: 1645 images (70.0%) [329 per class]
  Val:    352 images (15.0%) [~70 per class]
  Test:   353 images (15.0%) [~71 per class]
```

This creates `data/splits/train.csv`, `val.csv`, `test.csv` — pointer files that do **not** copy images (saves disk space).

---

## Step 2: CUDA / GPU Check

Before training, confirm your GPU is available inside the new `.venv`:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

> [!WARNING]
> `pip install torch` from PyPI may install the **CPU-only** build. If CUDA returns `False` but you have an NVIDIA GPU, reinstall with the CUDA wheel:
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
> ```

---

## Step 3: Train the Candidate Models

Train each architecture one at a time. Start with the lightest model first to validate the pipeline quickly.

### 3.1 — Recommended Training Order

| Order | Model | Reason |
| :---: | :--- | :--- |
| 1st | `mobilenet_v2` | Fastest to train (~minutes on GPU). Good sanity check. |
| 2nd | `efficientnet_b0` | Often best accuracy/speed trade-off. |
| 3rd | `densenet121` | Strongest for fine-grained leaf patterns. |
| 4th | `resnet50` | Standard baseline used in literature. |
| 5th | `vgg16` | Heaviest; train last for comparison. |

### 3.2 — Training Commands

All commands should be run with the `.venv` active:

```powershell
# Train MobileNetV2 (fast sanity check)
python train.py --model mobilenet_v2 --epochs 20

# Train EfficientNet-B0
python train.py --model efficientnet_b0 --epochs 20

# Train DenseNet121
python train.py --model densenet121 --epochs 20

# Train ResNet50
python train.py --model resnet50 --epochs 20

# Train VGG16
python train.py --model vgg16 --epochs 20
```

### 3.3 — What Gets Saved After Each Run

Each training run automatically produces:

| Output File | Location | Description |
| :--- | :--- | :--- |
| `best_<model>.pth` | `outputs/checkpoints/` | Best model weights (saved on best Macro F1) |
| `history_<model>.json` | `outputs/metrics/` | Per-epoch loss, accuracy, macro F1 log |
| `curves_<model>.png` | `outputs/plots/` | Training & validation loss/F1 curve plots |

> [!NOTE]
> Checkpointing is based on **Macro F1**, not accuracy. This prevents the model from "cheating" by being great at the majority classes and ignoring *Bud Rot*.

### 3.4 — Key Config Settings (tune if needed)

Open `config.py` to adjust:

| Setting | Current Value | What it controls |
| :--- | :---: | :--- |
| `NUM_EPOCHS` | `20` | Increase to 30–50 for real training |
| `BATCH_SIZE` | `32` | Reduce to `16` if GPU runs out of memory |
| `LEARNING_RATE` | `1e-4` | Standard Adam LR; can try `3e-4` |
| `LOSS_TYPE` | `weighted_ce` | Try `focal` for harder minority focus |
| `USE_BALANCED_SAMPLER` | `True` | Keep `True` — critical for imbalance |

---

## Step 4: Evaluate & Compare All Models

Once all models are trained, generate a comprehensive comparison report:

```powershell
# Evaluate a single model on the test set
python evaluate.py --model mobilenet_v2

# Generate full benchmark table for all trained models
python evaluate.py --benchmark_all
```

### 4.1 — Key Metrics to Focus On

> [!IMPORTANT]
> Do **NOT** use raw accuracy as your primary metric. The dataset is imbalanced. Use these instead:

| Metric | Why It Matters Here |
| :--- | :--- |
| **Macro F1-Score** | Treats all 5 classes equally — exposes failures on *Bud Rot* |
| **Per-class Recall for Bud Rot** | Clinical recall — missing this disease is the worst outcome |
| **Confusion Matrix** | Reveals which diseases are being confused with each other |
| **Latency (ms/img)** | Needed for edge deployment feasibility (MobileNetV2 target) |

### 4.2 — Baseline Expectations

For reference, reasonable performance benchmarks on the real dataset:

| Model | Expected Macro F1 (Real Data) | Inference Latency (GPU) |
| :--- | :---: | :---: |
| MobileNetV2 | ~80–85% | ~3–5 ms |
| EfficientNet-B0 | ~83–88% | ~5–8 ms |
| DenseNet121 | ~85–90% | ~10–15 ms |
| ResNet50 | ~82–87% | ~8–12 ms |
| VGG16 | ~80–84% | ~15–25 ms |

---

## Step 5: Generate Explainability Visualizations (Grad-CAM)

Once at least one model is trained on real data, generate Grad-CAM heatmaps to verify the model is learning disease-relevant features (leaf lesions, stem discoloration) and not background artifacts.

### 5.1 — Single Image Diagnosis with Explanation

```powershell
python predict.py --image "data/raw/Bud Rot/some_image.jpg" --explain
```

This outputs:
- **Top-3 class predictions** with confidence scores.
- **Grad-CAM overlay PNG** showing the region of the leaf/bark that drove the classification.

### 5.2 — Batch Inference on a Directory

```powershell
python predict.py --dir "data/raw/Gray Leaf Spot/" --output_csv results.csv
```

### 5.3 — What Good Grad-CAM Looks Like

| Disease | Expected Activation Region |
| :--- | :--- |
| Gray Leaf Spot | Circular necrotic spots on leaf blade |
| Leaf Rot | Distal leaf tip blackening region |
| Stem Bleeding | Bark/trunk area with reddish exudate marks |
| Bud Rot | Central spindle leaf and crown |
| Bud Root Dropping | Base of nut stalks and crown junction |

> [!TIP]
> If Grad-CAM activates on **background or irrelevant regions**, consider: adding more augmentation, increasing dropout, or using test-time augmentation.

---

## Step 6: Write the BTP Report (After Training)

With all experiments complete, your report should cover:

- [ ] **Abstract** — Problem, approach, key result (best Macro F1).
- [ ] **Literature Review** — Thite et al. (2023) baseline, imbalance strategies comparison.
- [ ] **Methodology** — Architecture choices, imbalance mitigation triple-strategy, evaluation protocol.
- [ ] **Results** — Comparison table (all 5 models × Macro F1, per-class F1, latency).
- [ ] **Explainability Analysis** — Grad-CAM visualizations for each class.
- [ ] **Conclusion** — Best model recommendation with deployment feasibility notes.

---

## Quick Reference — Command Cheat Sheet

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Step 1: Load real data (after placing images in data/raw/)
python prepare_data.py --split

# Step 2: Quick GPU check
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Step 3: Train models
python train.py --model mobilenet_v2 --epochs 20
python train.py --model efficientnet_b0 --epochs 20
python train.py --model densenet121 --epochs 20
python train.py --model resnet50 --epochs 20

# Step 4: Evaluate & benchmark
python evaluate.py --benchmark_all

# Step 5: Generate Grad-CAM explanation
python predict.py --image path/to/image.jpg --explain
```
