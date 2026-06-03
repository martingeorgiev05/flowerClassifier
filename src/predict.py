import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
from config import IMG_SIZE, MODEL_SAVE_PATH

CLASS_NAMES = tfds.builder('oxford_flowers102').info.features['label'].names

def load_model():
    return tf.keras.models.load_model(MODEL_SAVE_PATH)

def _run_inference(pil_img, model):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(np.array(img), axis=0)
    predictions = model.predict(img_array, verbose=0)
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
