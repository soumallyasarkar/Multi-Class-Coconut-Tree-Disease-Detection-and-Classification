"""
Central Configuration for Multi-Class Coconut Tree Disease Detection & Classification
B.Tech Project - IIIT Kottayam
"""

from pathlib import Path
import torch

# ==========================================
# 1. Project & File Paths
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SPLIT_DATA_DIR = DATA_DIR / "splits"

OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Ensure runtime directories exist
for path in [DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, METRICS_DIR, PLOTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Disease Categories & Benchmark Distribution (Thite et al., 2023)
# ==========================================
CLASSES = [
    "Bud Root Dropping",
    "Bud Rot",
    "Gray Leaf Spot",
    "Leaf Rot",
    "Stem Bleeding",
]
NUM_CLASSES = len(CLASSES)

# Original class counts from the paper (total: 5,798 images)
BENCHMARK_CLASS_COUNTS = {
    "Bud Root Dropping": 514,    # 8.9%  (Minority)
    "Bud Rot": 470,              # 8.1%  (Minority)
    "Gray Leaf Spot": 2135,      # 36.8% (Majority)
    "Leaf Rot": 1673,            # 28.9% (Majority)
    "Stem Bleeding": 1006,       # 17.3% (Moderate)
}

# ==========================================
# 3. Training & Hardware Hyperparameters
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# Image resolution: 256x256 retains fine-grained foliar necrosis details
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 3

# Safe worker count for Windows OS (avoids multiprocessing spawn deadlocks)
NUM_WORKERS = 2

# Train / Validation / Test split fractions
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==========================================
# 4. Imbalance Handling & Balanced Dataset Sampling
# ==========================================
# Balanced undersampling cap: set to None to utilize all 5,798 available images,
# while USE_BALANCED_SAMPLER dynamically ensures equal mini-batch representation.
SAMPLES_PER_CLASS = None

# Options for LOSS_TYPE: 'standard_ce', 'weighted_ce', 'focal'
LOSS_TYPE = "weighted_ce"

# Focal loss focusing parameter (gamma=2.0 is the standard setting in literature)
FOCAL_GAMMA = 2.0

# When True, dynamically draws mini-batches with equal class representation
USE_BALANCED_SAMPLER = True

# Candidate models supported by the framework
CANDIDATE_MODELS = [
    "mobilenet_v2",   # Ultra-lightweight for mobile / edge deployment
    "resnet50",       # Standard robust baseline in plant pathology literature
    "efficientnet_b0",# High-efficiency compound scaling
    "densenet121",    # Strong feature reuse for subtle leaf patterns
]

