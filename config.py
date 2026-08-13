"""
DICE-FER Configuration File
All hyperparameters and paths defined here.
"""

import os
import torch
# 
class Config:
    # ============ PATHS ============
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
    CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
    FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

    # ✅ FIXED: Point to actual dataset location
    DATA_ROOT = '.'
    DATASET_NAME = 'CK+48'

    # ============ MODEL ============
    BACKBONE = 'resnet18'
    FEATURE_DIM = 64
    IMAGE_SIZE = 112
    NUM_CHANNELS = 1
    NUM_EXPRESSIONS = 7

    # ============ TRAINING ============
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.0

    # Loss coefficients
    MU_EXP = 0.5
    NU_EXP = 1.0
    DELTA = 0.1
    MU_ID = 0.5
    NU_ID = 1.0
    ZETA_ADV = 0.025

    # ============ DEVICE ============
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = False  # ✅ CRITICAL: Disable AMP - causes NaN/Inf with MI losses
    NUM_WORKERS = 0  # ✅ Windows safe: No multiprocessing
    PIN_MEMORY = False  # ✅ Windows safe: Disable pinning

    # ============ EVALUATION ============
    TOP_K_RETRIEVAL = 5

    @classmethod
    def create_dirs(cls):
        """Create necessary directories."""
        for dir_path in [cls.RESULTS_DIR, cls.CHECKPOINT_DIR,
                         cls.FIGURES_DIR]:
            os.makedirs(dir_path, exist_ok=True)

    @classmethod
    def print_config(cls):
        """Print all configuration parameters."""
        print("=" * 60)
        print("DICE-FER Configuration")
        print("=" * 60)
        attrs = {k: v for k, v in vars(cls).items()
                 if not k.startswith('_') and not callable(v)}
        for key, value in sorted(attrs.items()):
            print(f"  {key}: {value}")
        print("=" * 60)