"""Compares the saved model on the validation set with abd without test-time augmentation (TTA)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf

from config import RESULTS_DIR
from src.data_loader import load_data
from src.evaluate import predict_dataset, report
from src.predict import load_model

out_dir = os.path.join(RESULTS_DIR, "07_tta")
model = load_model()

_, val_ds, _, class_names = load_data()
# Validation labels are one-hot (for training); the report expects plain class numbers
val_ds = val_ds.map(lambda images, labels: (images, tf.argmax(labels, axis=1)))

# One pass on the original photos, one on mirrored photos; TTA is their average
y_true, plain = predict_dataset(model, val_ds, tta=False)
mirrored_ds = val_ds.map(
    lambda images, labels: (tf.image.flip_left_right(images), labels)
)
_, mirrored = predict_dataset(model, mirrored_ds, tta=False)

report(y_true, plain, class_names, {"mode": "no_tta"}, out_dir, split="validation")
report(
    y_true,
    (plain + mirrored) / 2,
    class_names,
    {"mode": "tta"},
    out_dir,
    split="validation",
)
