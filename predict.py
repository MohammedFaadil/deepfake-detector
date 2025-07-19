'''////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) GAHRANOX INFOSEC 2025
//
// @Author: Mohammed Faadil
//
// Purpose: This script performs deepfake image classification using multiple pre-trained CNN models.
// It supports the following functionalities:
// 1) Loads model weights from disk (ResNet50, EfficientNetB4, InceptionV3, MobileNetV2, Xception)
// 2) Preprocesses image input to required format
// 3) Runs predictions and prints or returns classification results (FAKE or REAL)
//
// Remarks:
// - Ensure the trained model weights (.weights.h5) exist in the 'trained_models' directory.
// - Prediction threshold is set at 0.5: values >= 0.5 are classified as FAKE.
//
// Usage:
// python predict.py <path_to_image>
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////'''
# region Imports and Paths

import sys
import os
# Add root directory to path so we can import `predict.py`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Ensure we can import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from models import ResNet50, EfficientNetB4, InceptionV3, MobileNetV2, Xception

MODEL_BUILDERS = {
    "ResNet50": ResNet50.build_model,
    "EfficientNetB4": EfficientNetB4.build_model,
    "InceptionV3": InceptionV3.build_model,
    "MobileNetV2": MobileNetV2.build_model,
    "Xception": Xception.build_model
}

#MODEL_DIR = r"D:\hckply\conf\new deep fake\trained_models"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "trained_models")
IMG_SIZE = (224, 224)

def load_all_models():
    models = {}
    for name, builder in MODEL_BUILDERS.items():
        weight_path = os.path.join(MODEL_DIR, f"{name}.weights.h5")
        if os.path.exists(weight_path):
            model = builder(input_shape=(224, 224, 3))
            model.load_weights(weight_path)
            models[name] = model
            print(f"✅ Loaded {name}")
        else:
            print(f"❌ Weights not found for {name}")
    return models

def preprocess_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

def predict(models, img_path):
    img = preprocess_image(img_path)
    print(f"\n🔍 Predictions for: {os.path.basename(img_path)}")
    for name, model in models.items():
        prob = model.predict(img)[0][0]
        label = "FAKE" if prob >= 0.5 else "REAL"
        print(f"📌 {name}: {label} ({prob:.4f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide the path to an image file.")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        sys.exit(1)

    models = load_all_models()
    predict(models, image_path)


def predict_deepfake(image_path):
    models = load_all_models()
    img = preprocess_image(image_path)

    results = {}
    for name, model in models.items():
        prob = model.predict(img)[0][0]
        label = "FAKE" if prob >= 0.5 else "REAL"
        results[name] = {
            "label": label,
            "confidence": float(prob)
        }
    return results
