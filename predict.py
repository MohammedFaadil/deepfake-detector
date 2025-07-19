'''////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) GAHRANOX INFOSEC 2025
//
// @Author: Mohammed Faadil
//
// Purpose: Deepfake classification using CNNs with weights loaded from disk.
//
// Usage:
//   - CLI:   python predict.py <path_to_image>
//   - Module: from predict import predict_deepfake
////////////////////////////////////////////////////////////////////////////////////////////////////////'''

# region Imports and Setup
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Add root path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from models import ResNet50, EfficientNetB4, InceptionV3, MobileNetV2, Xception

MODEL_BUILDERS = {
    "ResNet50": ResNet50.build_model,
    "EfficientNetB4": EfficientNetB4.build_model,
    "InceptionV3": InceptionV3.build_model,
    "MobileNetV2": MobileNetV2.build_model,
    "Xception": Xception.build_model
}

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "trained_models")
IMG_SIZE = (224, 224)

# endregion

# region Model Loader with Error Handling
def load_all_models():
    models = {}
    for name, builder in MODEL_BUILDERS.items():
        weight_path = os.path.join(MODEL_DIR, f"{name}.weights.h5")
        print(f"\n🧠 Loading {name} from: {weight_path}")
        if os.path.exists(weight_path):
            try:
                model = builder(input_shape=(224, 224, 3))
                model.load_weights(weight_path)
                models[name] = model
                print(f"✅ Loaded {name}")
            except Exception as e:
                print(f"❌ Failed to load weights for {name}: {e}")
        else:
            print(f"⚠️ Weights not found for: {name}")
    return models
# endregion

# region Image Preprocessing
def preprocess_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)
# endregion

# region Predict CLI
def predict(models, img_path):
    img = preprocess_image(img_path)
    print(f"\n🔍 Predictions for: {os.path.basename(img_path)}")
    for name, model in models.items():
        try:
            prob = model.predict(img)[0][0]
            label = "FAKE" if prob >= 0.5 else "REAL"
            print(f"📌 {name}: {label} ({prob:.4f})")
        except Exception as e:
            print(f"⚠️ Prediction failed for {name}: {e}")
# endregion

# region Predict Programmatically
def predict_deepfake(image_path):
    models = load_all_models()
    img = preprocess_image(image_path)

    results = {}
    for name, model in models.items():
        try:
            prob = model.predict(img)[0][0]
            label = "FAKE" if prob >= 0.5 else "REAL"
            results[name] = {
                "label": label,
                "confidence": float(prob)
            }
        except Exception as e:
            results[name] = {
                "label": "ERROR",
                "confidence": None,
                "error": str(e)
            }
    return results
# endregion

# region CLI Entry Point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        sys.exit(1)

    models = load_all_models()
    predict(models, image_path)
# endregion
