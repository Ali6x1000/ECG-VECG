import os
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from google.colab import drive

from data import DataManager, CustomDataGenerator
from stats import StatsCalculator

# Mount Google Drive
drive.mount('/content/drive')

# ======================================================
# 0. Configuration
# ======================================================

# --- ❗️ CHOOSE YOUR EXPERIMENT ---
# Set this to 'ecg_only', 'vcg_only', or 'combined'
TRAINING_MODE = 'ecg_only' 
# ---

VAL_FOLD = 9
TEST_FOLD = 10
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DTYPE = np.float32

# --- File Paths ---
DRIVE_BASE_PATH = "/content/drive/MyDrive/CNN"

# --- Model and Plot Paths ---
BEST_MODEL_PATH = os.path.join(DRIVE_BASE_PATH, f"ecg_resnet50_{TRAINING_MODE}.keras")
PLOT_MAE_PATH = os.path.join(DRIVE_BASE_PATH, f"plot_{TRAINING_MODE}_mae.png")
PLOT_LOSS_PATH = os.path.join(DRIVE_BASE_PATH, f"plot_{TRAINING_MODE}_loss.png")

print(f"--- 🚀 Starting Experiment: {TRAINING_MODE} ---")

# ======================================================
# 1. Initialize Data Manager and Load Data
# ======================================================
data_manager = DataManager(TRAINING_MODE, DRIVE_BASE_PATH, IMG_SIZE, DTYPE)
data_manager.copy_data_from_drive()

Y_full, N_RECORDS = data_manager.load_metadata()

# ======================================================
# 2. Generate Image Cache
# ======================================================
data_manager.generate_image_cache(Y_full, N_RECORDS)

# ======================================================
# 3. Calculate Normalization Stats
# ======================================================
stats_calculator = StatsCalculator(TRAINING_MODE, DRIVE_BASE_PATH, IMG_SIZE)
channel_mean, channel_std = stats_calculator.calculate_or_load_stats(
    data_manager.local_image_cache_path, N_RECORDS)

# ======================================================
# 4. Prepare Data Splits and Generators
# ======================================================
y_labels = Y_full.age.values.astype(DTYPE)

# Get indices based on the full, unfiltered metadata
train_idx = np.where((Y_full.strat_fold != TEST_FOLD) & (Y_full.strat_fold != VAL_FOLD))[0]
val_idx = np.where(Y_full.strat_fold == VAL_FOLD)[0]
test_idx = np.where(Y_full.strat_fold == TEST_FOLD)[0]

# --- Instantiate generators ---
print("\n--- Initializing Data Generators ---")
train_generator = CustomDataGenerator(data_manager.local_image_cache_path, y_labels, train_idx, 
                                    BATCH_SIZE, channel_mean, channel_std, IMG_SIZE, N_RECORDS)
val_generator = CustomDataGenerator(data_manager.local_image_cache_path, y_labels, val_idx, 
                                  BATCH_SIZE, channel_mean, channel_std, IMG_SIZE, N_RECORDS)

# ======================================================
# 5. Model Building & Training
# ======================================================

def build_model():
    """Build ResNet50-based model for age prediction"""
    base_model = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    inputs = base_model.input
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(1, name="age_prediction")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model, base_model

print("\n🛠️ Building model...")
model, base_model = build_model()
print("✅ Model built.")

# --- PHASE 1: Train the Head Only ---
print("\n--- PHASE 1: Training the model head ---")
base_model.trainable = False # Freeze the base

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), # Use a higher LR for the head
    loss="mean_squared_error",
    metrics=["mean_absolute_error"]
)

history_head = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_mean_absolute_error", patience=5, restore_best_weights=True)
    ]
)
print("✅ Head training complete.")

# --- PHASE 2: Unfreeze and Fine-Tune Everything ---
print("\n--- PHASE 2: Fine-tuning the entire model ---")
base_model.trainable = True # Unfreeze the base

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), # CRITICAL: Use a very low LR
    loss="mean_squared_error",
    metrics=["mean_absolute_error"]
)

history_fine_tune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50, # Train for more epochs
    callbacks=[
        keras.callbacks.ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, monitor="val_mean_absolute_error", mode="min"),
        keras.callbacks.EarlyStopping(monitor="val_mean_absolute_error", patience=10, mode="min")
    ]
)
print("✅ Fine-tuning complete.")

# ======================================================
# 6. Final Evaluation
# ======================================================
def evaluate_model():
    """Evaluate the best model on the test set"""
    print("\n🧪 Evaluating the best fine-tuned model on the test set...")
    # Load the best weights saved by ModelCheckpoint
    model.load_weights(BEST_MODEL_PATH)

    # Create the test generator
    test_generator = CustomDataGenerator(data_manager.local_image_cache_path, y_labels, test_idx, 
                                       BATCH_SIZE, channel_mean, channel_std, IMG_SIZE, N_RECORDS)
    test_loss, test_mae = model.evaluate(test_generator)

    print("\n--- Evaluation Complete ---")
    print(f"Mode: {TRAINING_MODE}")
    print(f"📉 Final Test Loss: {test_loss:.4f}")
    print(f"🏆 Final Test MAE: {test_mae:.4f}")

evaluate_model()

# ======================================================
# 7. Plot and Save Training History
# ======================================================
def plot_training_history():
    """Generate and save training plots"""
    print("\n📊 Generating and saving training plots...")

    try:
        # Combine history from the two training phases
        history = {}
        history['mean_absolute_error'] = history_head.history['mean_absolute_error'] + history_fine_tune.history['mean_absolute_error']
        history['val_mean_absolute_error'] = history_head.history['val_mean_absolute_error'] + history_fine_tune.history['val_mean_absolute_error']
        history['loss'] = history_head.history['loss'] + history_fine_tune.history['loss']
        history['val_loss'] = history_head.history['val_loss'] + history_fine_tune.history['val_loss']

        # --- Plot MAE ---
        plt.figure(figsize=(10, 6))
        plt.plot(history['mean_absolute_error'], label='Training MAE')
        plt.plot(history['val_mean_absolute_error'], label='Validation MAE')
        plt.axvline(x=len(history_head.history['loss'])-1, color='gray', linestyle='--', label='Start Fine-Tuning')
        plt.title(f'Training and Validation MAE ({TRAINING_MODE})')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error (Age)')
        plt.legend()
        plt.grid(True)
        plt.savefig(PLOT_MAE_PATH)
        print(f"✅ MAE plot saved to {PLOT_MAE_PATH}")
        plt.close()

        # --- Plot Loss ---
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.axvline(x=len(history_head.history['loss'])-1, color='gray', linestyle='--', label='Start Fine-Tuning')
        plt.title(f'Training and Validation Loss ({TRAINING_MODE})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig(PLOT_LOSS_PATH)
        print(f"✅ Loss plot saved to {PLOT_LOSS_PATH}")
        plt.close()

    except Exception as e:
        print(f"⚠️ Error saving plots: {e}")

plot_training_history()

print("\n--- 🎉 Experiment Finished ---")