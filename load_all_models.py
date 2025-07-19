'''////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) GAHRANOX INFOSEC 2025
//
// @Author: Mohammed Faadil
//
// Purpose: This module provides functionality to load pre-trained deep learning models
// from disk using corresponding model builders (ResNet50, EfficientNetB4, InceptionV3, etc.).
// 1) Dynamically matches each .weights.h5 file with its corresponding model builder.
// 2) Loads and compiles each model from the stored weights.
// 3) Returns a dictionary of all successfully loaded models for downstream inference or evaluation.
//
// Remarks:
// - Make sure the weights directory and model names match exactly.
// - All model builders must expose a `build_model(input_shape=...)` method.
// - This module assumes models were trained using consistent architecture definitions.
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////'''

import os
from models import ResNet50, EfficientNetB4, InceptionV3, MobileNetV2, Xception
MODEL_BUILDERS = {
    "ResNet50": ResNet50.build_model,
    "EfficientNetB4": EfficientNetB4.build_model,
    "InceptionV3": InceptionV3.build_model,
    "MobileNetV2": MobileNetV2.build_model,
    "Xception": Xception.build_model
}

def load_models(path='trained_models'):
    models = {}
    for file in os.listdir(path):
        if file.endswith('.weights.h5'):
            name = file.replace('.weights.h5', '')
            builder = MODEL_BUILDERS.get(name)
            if builder:
                try:
                    model = builder(input_shape=(224, 224, 3))
                    model.load_weights(os.path.join(path, file))
                    models[name] = model
                    print(f"✅ Loaded model: {name}")
                except Exception as e:
                    print(f"❌ Failed to load {name}: {e}")
    return models
