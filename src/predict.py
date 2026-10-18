import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

from config import MODEL_SAVE_PATH, USE_TTA
from src.preprocessing import preprocess_image

CLASS_NAMES = tfds.builder("oxford_flowers102").info.features["label"].names


def load_model():
    return tf.keras.models.load_model(MODEL_SAVE_PATH, compile=False)


def predict_probs(model, images, tta=USE_TTA):
    """Class probabilities for a batch of preprocessed images."""

    probs = model.predict(images, verbose=0)
    if tta:
        probs = (probs + model.predict(tf.image.flip_left_right(images), verbose=0)) / 2

    return probs


def _run_inference(pil_img, model):
    img = preprocess_image(np.array(pil_img.convert("RGB")))
    probs = predict_probs(model, img[tf.newaxis])[0]
    idx = int(np.argmax(probs))
    confidence = round(float(probs[idx] * 100), 2)
    return CLASS_NAMES[idx], confidence


def predict(image_path, model=None):
    if model is None:
        model = load_model()
    img = Image.open(image_path)
    return _run_inference(img, model)


def predict_from_pil(pil_img, model):
    return _run_inference(pil_img, model)
