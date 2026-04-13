# src/predict.py
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
from config import IMG_SIZE, MODEL_SAVE_PATH

CLASS_NAMES = tfds.builder('oxford_flowers102').info.features['label'].names

def load_model():
    return tf.keras.models.load_model(MODEL_SAVE_PATH)

def predict(image_path, model=None):
    if model is None:
        model = load_model()

    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(np.array(img), axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index] * 100)

    flower = CLASS_NAMES[predicted_index]
    print(f"\n🌸 Predicted: {flower.upper()}  ({confidence:.1f}% confidence)")
    return flower, confidence