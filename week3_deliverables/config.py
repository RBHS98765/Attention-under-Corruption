"""Week 3 configuration freeze (Part E).

The architecture is fixed here and is NOT changed per attention method. This
file is the single source of truth for the training and evaluation pipeline.
"""
import os

# ---------------------------------------------------------------------------
# Architecture (frozen from Week 2)
# ---------------------------------------------------------------------------
BACKBONE_NAME = "MatchedBackbone"
INPUT_SIZE = [3, 32, 32]
NUM_CLASSES = 10
CHANNEL_WIDTHS = [64, 64, 128, 256]
NUM_RESIDUAL_BLOCKS = 4
RESIDUAL_BLOCKS_PER_STAGE = [2, 1, 1]
ATTENTION_INSERTION = "after_conv_transformation_before_residual_add (every block)"
ATTENTION_TYPES = ["none", "se", "bam", "cbam"]

SE_REDUCTION = 16
BAM_SETTINGS = {"reduction_ratio": 16, "dilation_conv_num": 2, "dilation_val": 4}
CBAM_SETTINGS = {"reduction_ratio": 16, "pool_types": ["avg", "max"], "no_spatial": False}

NORMALIZATION = "BatchNorm2d"
ACTIVATION = "ReLU"
CLASSIFIER = "AdaptiveAvgPool2d(1) -> Linear(256, 10)"

# ---------------------------------------------------------------------------
# Training (Part B)
# ---------------------------------------------------------------------------
OPTIMIZER = "SGD"
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LR_SCHEDULE = "cosine_annealing"
BATCH_SIZE = 128
EPOCHS = 1  # Week 3 smoke test: at most one epoch
TRAIN_SUBSET = 10000  # smoke test uses 10,000 training images
AUGMENTATION = ["RandomCrop(32, padding=4)", "RandomHorizontalFlip", "ToTensor", "Normalize(mean, std)"]
NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
NORMALIZE_STD = (0.2023, 0.1994, 0.2010)

# ---------------------------------------------------------------------------
# Determinism / paths (Parts B2, B3, B4)
# ---------------------------------------------------------------------------
SEED = 0
SEED_METHOD = "set_seed(): random, numpy, torch CPU, torch CUDA (if any); cudnn.deterministic=True, cudnn.benchmark=False"
NUM_THREADS = 8
EVAL_CLEAN_SUBSET = 1000   # smoke-test eval: first N clean test images
EVAL_CORRUPTION_SUBSET = 500  # smoke-test eval: first N images per severity
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
DATA_DIR = "/home/daytona/work/data"
CKPT_NAME = "ckpt_{variant}_{seed}_epoch_{epoch}.pt"  # never overwrite across seeds

# ---------------------------------------------------------------------------
# CIFAR-10-C evaluation (Part C)
# ---------------------------------------------------------------------------
PRIMARY_CORRUPTIONS = ["brightness", "contrast", "defocus_blur", "elastic_transform"]
SEVERITIES = [1, 2, 3, 4, 5]
CIFAR10C_DIR = "/home/daytona/work/data/cifar10-c"
CLEAN_SEVERITY = 0  # clean CIFAR-10 test set is severity 0
