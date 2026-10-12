"""Saves a grid of augmented versions of one training image."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from config import IMG_SIZE, RESULTS_DIR
from src.model import build_augmentation

out_name = sys.argv[1] if len(sys.argv) > 1 else "augmentation_preview.png"

ds = tfds.load("oxford_flowers102", split="train", as_supervised=True)
image, _ = next(iter(ds.skip(3)))
image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))[tf.newaxis]

augment = build_augmentation()
plt.figure(figsize=(12, 6))
for i in range(8):
    out = augment(image, training=True)[0]
    plt.subplot(2, 4, i + 1)
    plt.imshow(tf.cast(tf.clip_by_value(out, 0, 255), tf.uint8))
    plt.axis("off")
plt.tight_layout()
Path(RESULTS_DIR).mkdir(exist_ok=True)
plt.savefig(Path(RESULTS_DIR) / out_name)
print(f"Saved {RESULTS_DIR}/{out_name}")
