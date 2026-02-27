import os
import numpy as np
import pandas as pd
import cv2
from scipy import signal
from tqdm import tqdm
import keras
import tensorflow as tf

# ======================================================
# 0. Configuration
# ======================================================
VAL_FOLD = 9
TEST_FOLD = 10
IMG_SIZE = (224, 224)
BATCH_SIZE = 32 # Increased batch size for potentially faster training
N_RECORDS = 21799
DTYPE = np.float32

# --- File Paths ---
DRIVE_BASE_PATH = "/content/drive/MyDrive/CNN"
RAW_CACHE_PATH = "./ptbxl_image_cache_full.npy"
META_CSV_PATH = os.path.join(DRIVE_BASE_PATH, "ptbxl_database.csv")
BEST_MODEL_PATH = os.path.join(DRIVE_BASE_PATH, "ecg_resnet50_finetuned.keras")

# --- New Cache Paths for ECG-only Images ---
IMAGE_CACHE_PATH = os.path.join(DRIVE_BASE_PATH, "ptbxl_ecg_only_image_cache.npy")
STATS_PATH = os.path.join(DRIVE_BASE_PATH, "ecg_only_image_stats.npz")
LOCAL_IMAGE_CACHE_PATH = "/content/ptbxl_image_cache_full.npy"

# Copy data from Drive to local Colab disk for speed
if os.path.exists(DRIVE_BASE_PATH):
    print("🚚 Copying data from Google Drive to local disk...")
    # Use standard copy, as rsync isn't a native python command
    from shutil import copyfile
    if os.path.exists(IMAGE_CACHE_PATH):
        copyfile(IMAGE_CACHE_PATH, LOCAL_IMAGE_CACHE_PATH)
        print("✅ Copy complete!")
    else:
        print("⚠️ Image cache not found on Drive. It will be generated locally.")
else:
    print("⚠️ Google Drive not mounted or path is incorrect. Using local paths.")

# ======================================================
# 1. Preprocessing: Image Generation (ECG-Only)
# ======================================================

def create_grayscale_spectrogram(signal_1d, fs=500):
    nperseg = 32
    noverlap = 28
    _, _, Sxx = signal.spectrogram(signal_1d, fs=fs, nperseg=nperseg, noverlap=noverlap)
    if Sxx.shape[1] == 0: return np.zeros(IMG_SIZE, dtype=DTYPE)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    Sxx_norm = (Sxx_db - np.min(Sxx_db)) / (np.max(Sxx_db) - np.min(Sxx_db) + 1e-10)
    resized_spec = cv2.resize(Sxx_norm, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    return resized_spec.astype(DTYPE)

if not os.path.exists(LOCAL_IMAGE_CACHE_PATH):
    print("🛠️ ECG image cache not found. Generating and caching all images now...")
    X_ecg_raw = np.memmap(RAW_CACHE_PATH, dtype=DTYPE, mode='r', shape=(N_RECORDS, 5000, 12))
    image_cache = np.memmap(LOCAL_IMAGE_CACHE_PATH, dtype=DTYPE, mode='w+', shape=(N_RECORDS, *IMG_SIZE, 3))

    for i in tqdm(range(N_RECORDS), desc="Generating ECG Images"):
        ecg_signal = X_ecg_raw[i]
        ecg_specs = [create_grayscale_spectrogram(ecg_signal[:, j]) for j in range(12)]
        ecg_img_gray = np.mean(ecg_specs, axis=0)
        # Stack the single channel 3 times to create an RGB-like image for ResNet
        image_cache[i] = np.stack([ecg_img_gray, ecg_img_gray, ecg_img_gray], axis=-1)
    image_cache.flush()
    print("✅ All ECG images have been generated and cached.")
else:
    print("✅ ECG image cache found. Skipping image generation.")


# ======================================================
# 2. Preprocessing: Calculate Normalization Stats
# ======================================================
if not os.path.exists(STATS_PATH):
    print("🛠️ Global stats not found. Calculating stats from image cache...")
    image_cache = np.memmap(LOCAL_IMAGE_CACHE_PATH, dtype=DTYPE, mode='r', shape=(N_RECORDS, *IMG_SIZE, 3))
    
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    num_pixels = N_RECORDS * IMG_SIZE[0] * IMG_SIZE[1]

    for i in tqdm(range(0, N_RECORDS, 256), desc="Calculating Stats"):
        chunk = image_cache[i:i+256].astype(np.float64)
        pixel_sum += np.sum(chunk, axis=(0, 1, 2))
        pixel_sq_sum += np.sum(chunk**2, axis=(0, 1, 2))

    channel_mean = pixel_sum / num_pixels
    channel_std = np.sqrt(pixel_sq_sum / num_pixels - channel_mean**2)
    
    np.savez(STATS_PATH, mean=channel_mean, std=channel_std)
    print(f"✅ Stats calculated and saved.")
else:
    stats = np.load(STATS_PATH)
    channel_mean = stats['mean']
    channel_std = stats['std']
    print(f"✅ Per-channel stats loaded.")

# ======================================================
# 3. Data Generator (with In-Memory Caching)
# ======================================================
class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, image_cache_path, labels, indices, batch_size, mean, std):
        self.indices = indices
        self.batch_size = batch_size
        self.mean = mean
        self.std = std

        print(f"🧠 Preloading {len(indices)} samples into RAM...")
        image_data_memmap = np.memmap(image_cache_path, dtype=np.float32, mode='r', shape=(N_RECORDS, *IMG_SIZE, 3))
        self.X_cached = image_data_memmap[indices]
        self.y_cached = labels[indices]
        print("✅ Preloading complete.")

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.indices))
        
        X_batch = self.X_cached[start:end]
        y_batch = self.y_cached[start:end]

        X_batch = (X_batch - self.mean) / (self.std + 1e-7)
        return X_batch, y_batch

# ======================================================
# 4. Model Building & Training
# ======================================================

# --- Load metadata and define splits ---
Y = pd.read_csv(META_CSV_PATH, index_col="ecg_id")
y_labels = Y.age.values.astype(DTYPE)
train_idx = np.where((Y.strat_fold != TEST_FOLD) & (Y.strat_fold != VAL_FOLD))[0]
val_idx = np.where(Y.strat_fold == VAL_FOLD)[0]
test_idx = np.where(Y.strat_fold == TEST_FOLD)[0]

# --- Instantiate generators ---
train_generator = CustomDataGenerator(LOCAL_IMAGE_CACHE_PATH, y_labels, train_idx, BATCH_SIZE, channel_mean, channel_std)
val_generator = CustomDataGenerator(LOCAL_IMAGE_CACHE_PATH, y_labels, val_idx, BATCH_SIZE, channel_mean, channel_std)
test_generator = CustomDataGenerator(LOCAL_IMAGE_CACHE_PATH, y_labels, test_idx, BATCH_SIZE, channel_mean, channel_std)

# --- Build the model with a new head ---
print("\n🛠️ Building model...")
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
print("✅ Model built.")

# --- PHASE 1: Train the Head Only ---
print("\n--- PHASE 1: Training the model head ---")
base_model.trainable = False # Freeze the base

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), # Use a higher LR for the head
    loss="mean_squared_error",
    metrics=["mean_absolute_error"]
)

# Train for a few epochs to let the new head learn
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
# 5. Final Evaluation
# ======================================================
print("\n🧪 Evaluating the best fine-tuned model on the test set...")
# Keras automatically keeps the best weights from EarlyStopping/ModelCheckpoint
model.load_weights(BEST_MODEL_PATH)
test_loss, test_mae = model.evaluate(test_generator)

print("\n--- Evaluation Complete ---")
print(f"📉 Final Test Loss: {test_loss:.4f}")
print(f"🏆 Final Test MAE: {test_mae:.4f}")