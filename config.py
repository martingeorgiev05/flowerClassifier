import os


# Image settings
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 20       
NUM_CLASSES = 102       
LEARNING_RATE = 0.001

# Class names
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "flower_photos")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "flower_model.h5")

# Training settings
VALIDATION_SPLIT = 0.2
SEED = 123
LEARNING_RATE = 0.001