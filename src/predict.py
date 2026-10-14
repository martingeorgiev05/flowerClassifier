import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

from config import MODEL_SAVE_PATH
from src.preprocessing import preprocess_image

CLASS_NAMES = tfds.builder("oxford_flowers102").info.features["label"].names


def load_model():
    return tf.keras.models.load_model(MODEL_SAVE_PATH, compile=False)


def _run_inference(pil_img, model):
    img = preprocess_image(np.array(pil_img.convert("RGB")))
    predictions = model.predict(img[tf.newaxis], verbose=0)
    idx = int(np.argmax(predictions[0]))
    confidence = round(float(predictions[0][idx] * 100), 2)
    return CLASS_NAMES[idx], confidence


def predict(image_path, model=None):
    if model is None:
        model = load_model()
    img = Image.open(image_path)
    return _run_inference(img, model)


def predict_from_pil(pil_img, model):
    return _run_inference(pil_img, model)
