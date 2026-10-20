"""Saves a slim copy of the trained model (without optimizer state) for deployment."""

import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
from PIL import Image

from config import BASE_DIR, MODEL_SAVE_PATH
from src.preprocessing import preprocess_image

SLIM_PATH = os.path.join(BASE_DIR, "models", "flower_model_slim.h5")

model = tf.keras.models.load_model(MODEL_SAVE_PATH, compile=False)
model.save(SLIM_PATH, include_optimizer=False)
slim = tf.keras.models.load_model(SLIM_PATH, compile=False)

# Check that both files predict exactly the same on the sample photos
paths = sorted(glob.glob(os.path.join(BASE_DIR, "sample_images", "*.jpg")))
batch = np.stack(
    [preprocess_image(np.array(Image.open(p).convert("RGB"))).numpy() for p in paths]
)
original_probs = model.predict(batch, verbose=0)
slim_probs = slim.predict(batch, verbose=0)
assert np.allclose(original_probs, slim_probs, atol=1e-6), (
    "Slim model predicts differently!"
)

print(f"Original {os.path.getsize(MODEL_SAVE_PATH) / 1e6:.0f} MB")
print(f"Slim: {os.path.getsize(SLIM_PATH) / 1e6:.0f} MB -> {SLIM_PATH}")
print(f"Verified: identical predictions on {len(paths)} sample images")
