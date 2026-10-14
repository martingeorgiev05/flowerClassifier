import os

# Image settings
IMG_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 102
LEARNING_RATE = 0.001

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "flower_model.h5")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Phase 2(fine-tuning)
FINE_TUNE_FROM = "block6a_expand_conv"
FINE_TUNE_EPOCHS = 30
FINE_TUNE_LR = 3e-5
FINE_TUNE_WARMUP_EPOCHS = 2

FINAL_PHASE1_EPOCHS = 30
FINAL_PHASE2_EPOCHS = 15

LABEL_SMOOTHING = 0.1
