import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict

image_path = "sample_images/daisy-test.jpg"
flower, confidence = predict(image_path)