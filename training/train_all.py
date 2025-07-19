'''////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) GAHRANOX INFOSEC 2025
//
// @Author: Mohammed Faadil
//
// Purpose: This script handles the training of deep learning image classification models.
// The models supported are ResNet50, EfficientNetB4, InceptionV3, MobileNetV2, and Xception.
// The script enables GPU memory growth, sets up data generators, compiles models with safe optimizer settings,
// includes NaN-safe callbacks, and stores trained model weights.
// 1) Load and preprocess image data
// 2) Dynamically select and build a model from predefined architectures
// 3) Train the model with callbacks (EarlyStopping, NaN monitor)
// 4) Save the final model weights for later inference or evaluation
// Remarks: Make sure the 'models' directory has each model's `build_model()` implemented.
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////'''


import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, Callback

# ✅ Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ GPU memory growth enabled.")
    except RuntimeError as e:
        print("⚠️ Could not set memory growth:", e)

# ✅ Import model builders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import ResNet50, EfficientNetB4, InceptionV3, MobileNetV2, Xception

# ✅ Available model map
MODELS = {
    "ResNet50": ResNet50.build_model,
    "EfficientNetB4": EfficientNetB4.build_model,
    "InceptionV3": InceptionV3.build_model,
    "MobileNetV2": MobileNetV2.build_model,
    "Xception": Xception.build_model
}

# ✅ Paths and configuration
DATA_DIR = 'training/datasets/static'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
BATCH_SIZE = 4
IMG_SIZE = (224, 224)
EPOCHS = 5
SAVE_PATH = 'trained_models'

# ✅ Custom callback for detecting NaN loss
class NaNStoppingCallback(Callback):
    def on_batch_end(self, batch, logs=None):
        loss = logs.get("loss")
        if loss is None or tf.math.is_nan(loss):
            print(f"❌ NaN loss detected at batch {batch}. Stopping training.")
            self.model.stop_training = True

# ✅ Optional debugging callback
class DebugCallback(Callback):
    def on_train_batch_end(self, batch, logs=None):
        loss = logs.get("loss")
        if loss is not None and tf.math.is_nan(loss):
            print("🧨 Detected NaN at batch", batch)

# ✅ Training function
def train_model(name, build_fn):
    print(f"\n🔧 Training model: {name}")

    # ✅ Image preprocessing
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    # ✅ Build model
    model = build_fn(input_shape=(224, 224, 3))

    # ✅ Compile with safe optimizer settings
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0,
        clipvalue=1.0
    )

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # ✅ Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        NaNStoppingCallback(),
        DebugCallback()  # Remove this line if you don't want debug logging
    ]

    # ✅ Start training
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # ✅ Save model weights
    os.makedirs(SAVE_PATH, exist_ok=True)
    model_path = os.path.join(SAVE_PATH, f"{name}.weights.h5")
    model.save_weights(model_path)
    print(f"✅ Model weights saved to {model_path}\n")

# ✅ Main execution
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\n❌ Usage: python train_all.py <ModelName>")
        print("   Available models:", ", ".join(MODELS.keys()))
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name not in MODELS:
        print(f"\n❌ Model '{model_name}' not found. Choose from: {', '.join(MODELS.keys())}")
        sys.exit(1)

    train_model(model_name, MODELS[model_name])
