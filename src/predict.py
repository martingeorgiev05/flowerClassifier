# src/predict.py
import tensorflow as tf
import numpy as np
from PIL import Image
from config import IMG_SIZE, CLASS_NAMES, MODEL_SAVE_PATH

def load_model():
    """Loads the saved model from disk."""
    return tf.keras.models.load_model(MODEL_SAVE_PATH)


def predict(image_path, model=None):
    """Takes an image path and returns the predicted flower name + confidence."""
    if model is None:
        model = load_model()

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(np.array(img), axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence      = predictions[0][predicted_index] * 100

    flower = CLASS_NAMES[predicted_index]
    print(f"\n Predicted: {flower.upper()}  ({confidence:.1f}% confidence)")
    return flower, confidence