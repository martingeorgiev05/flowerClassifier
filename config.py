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


FINAL_PHASE1_EPOCHS = 30
FINAL_PHASE2_EPOCHS = 15
