import os
import numpy as np
import pandas as pd
import cv2
from scipy import signal
from tqdm import tqdm

# Set the Keras backend to torch *before* importing keras and any related libraries
os.environ["KERAS_BACKEND"] = "torch"
import keras



# ======================================================
# 0. Configuration
# ======================================================
VAL_FOLD = 9
TEST_FOLD = 10
IMG_SIZE = (224, 224)
BATCH_SIZE = 20
N_RECORDS = 21799
DTYPE = np.float32


DRIVE_BASE_PATH = "/content/drive/MyDrive/CNN"

# --- New File Paths ---
RAW_CACHE_PATH = os.path.join(DRIVE_BASE_PATH, "ptbxl_500hz_cache.npy")
IMAGE_CACHE_PATH = os.path.join(DRIVE_BASE_PATH, "ptbxl_image_cache.npy") # For pre-generated images
STATS_PATH = os.path.join(DRIVE_BASE_PATH, "global_image_stats.npz") # For global mean/std
META_CSV_PATH = os.path.join(DRIVE_BASE_PATH, "ptbxl_database.csv")
BEST_MODEL_PATH = os.path.join(DRIVE_BASE_PATH, "ecg_resnet_v3.keras")

# Original path on your Google Drive
DRIVE_IMAGE_CACHE_PATH = os.path.join(DRIVE_BASE_PATH, "ptbxl_image_cache.npy")

# New path on the fast, local Colab disk
LOCAL_IMAGE_CACHE_PATH = "/content/ptbxl_image_cache_full.npy"

print("🚚 Copying data from Google Drive to local disk... (This may take a few minutes)")
# Use rsync for a progress bar and efficient copying
# !rsync -ah --progress {DRIVE_IMAGE_CACHE_PATH} {LOCAL_IMAGE_CACHE_PATH}
print("✅ Copy complete!")
# ======================================================
# 1. Preprocessing Step 1: Image Generation & Caching
# ======================================================

# --- Helper functions (used only during caching) ---
def derive_vcg_from_ecg(ecg_signals):
    I, II, V1, V2, V3, V4, V5, V6 = [ecg_signals[..., i] for i in [0, 1, 6, 7, 8, 9, 10, 11]]
    X = (0.38*I + 0.07*II - 0.13*V1 + 0.05*V2 - 0.01*V3 + 0.14*V4 + 0.06*V5 + 0.54*V6)
    Y = (0.11*I - 0.23*II + 0.06*V1 - 0.02*V2 + 0.05*V3 + 0.06*V4 - 0.17*V5 + 0.13*V6)
    Z = (-0.08*I + 0.06*II - 0.36*V1 + 0.14*V2 - 0.27*V3 + 0.21*V4 - 0.10*V5 + 0.06*V6)
    return np.stack([X, Y, Z], axis=-1)

def create_grayscale_spectrogram(signal_1d, fs=500):
    nperseg = 32
    noverlap = 28
    frequencies, times, Sxx = signal.spectrogram(signal_1d, fs=fs, nperseg=nperseg, noverlap=noverlap)
    if Sxx.shape[1] == 0: return np.zeros(IMG_SIZE, dtype=DTYPE)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    Sxx_norm = (Sxx_db - np.min(Sxx_db)) / (np.max(Sxx_db) - np.min(Sxx_db) + 1e-10)
    resized_spec = cv2.resize(Sxx_norm, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    return resized_spec.astype(DTYPE)

# --- Main Caching Logic ---
if not os.path.exists(IMAGE_CACHE_PATH):
    print("🛠️ Image cache not found. Generating and caching all images now...")
    print("This will take a long time but only needs to be done once.")

    X_ecg_raw = np.memmap(RAW_CACHE_PATH, dtype=DTYPE, mode='r', shape=(N_RECORDS, 5000, 12))
    image_cache = np.memmap(IMAGE_CACHE_PATH, dtype=DTYPE, mode='w+', shape=(N_RECORDS, *IMG_SIZE, 3))

    for i in tqdm(range(N_RECORDS), desc="Generating Images"):
        ecg_signal = X_ecg_raw[i:i+1] # Keep dimensions for VCG derivation
        vcg_signal = derive_vcg_from_ecg(ecg_signal)

        ecg_specs = [create_grayscale_spectrogram(ecg_signal[0, :, j]) for j in range(12)]
        ecg_img_gray = np.mean(ecg_specs, axis=0)

        vcg_specs = [create_grayscale_spectrogram(vcg_signal[0, :, j]) for j in range(3)]
        vcg_img_gray = np.mean(vcg_specs, axis=0)

        squeezed_img = (ecg_img_gray + vcg_img_gray) / 2.0
        
        image_cache[i] = np.stack([ecg_img_gray, vcg_img_gray, squeezed_img], axis=-1)
    
    image_cache.flush()
    print("✅ All images have been generated and cached.")
else:
    print("✅ Image cache found. Skipping image generation.")


# ======================================================
# 2. Preprocessing Step 2: Calculate  Normalization Stats
# ======================================================
if not os.path.exists(STATS_PATH):
    print("🛠️ Global stats file not found. Calculating per-channel stats from image cache...")
    image_cache = np.memmap(IMAGE_CACHE_PATH, dtype=DTYPE, mode='r', shape=(N_RECORDS, *IMG_SIZE, 3))
    
    # Calculate mean and std in chunks to conserve RAM
    chunk_size = 256
    
    # ✨ KEY CHANGE: Initialize sums as vectors of 3 (for R, G, B)
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    num_pixels_per_channel = N_RECORDS * IMG_SIZE[0] * IMG_SIZE[1]

    for i in tqdm(range(0, N_RECORDS, chunk_size), desc="Calculating Stats"):
        chunk = image_cache[i:i+chunk_size].astype(np.float64)
        
        # ✨ KEY CHANGE: Sum along all axes EXCEPT the channel axis
        pixel_sum += np.sum(chunk, axis=(0, 1, 2))
        pixel_sq_sum += np.sum(chunk**2, axis=(0, 1, 2))
        
    # The math is the same, but now it operates on vectors
    channel_mean = pixel_sum / num_pixels_per_channel
    channel_std = np.sqrt(pixel_sq_sum / num_pixels_per_channel - channel_mean**2)

    # Save the new vector stats
    np.savez(STATS_PATH, mean=channel_mean, std=channel_std)
    print(f"✅ Per-channel stats calculated and saved.")
    print(f"   Means (R, G, B): {channel_mean}")
    print(f"   Stds (R, G, B): {channel_std}")

else:
    stats = np.load(STATS_PATH)
    channel_mean = stats['mean']
    channel_std = stats['std']
    # Reshape for broadcasting in the generator
    print(f"✅ Per-channel stats loaded from file.")
    print(f"   Means (R, G, B): {channel_mean}")
    print(f"   Stds (R, G, B): {channel_std}")

# 3. Dataset Loading (Data Generator)
# ======================================================
def spec_augment(image, time_masking_para=40, frequency_masking_para=15):
    """Applies augmentation to each channel of the 3-channel image."""
    augmented_image = image.copy()
    height, width, _ = augmented_image.shape
    
    for i in range(3): # Apply to each channel
        # Frequency masking
        f_zero = int(np.random.uniform(0, frequency_masking_para))
        f_start = np.random.randint(0, height - f_zero)
        augmented_image[f_start:f_start + f_zero, :, i] = 0
        
        # Time masking
        t_zero = int(np.random.uniform(0, time_masking_para))
        t_start = np.random.randint(0, width - t_zero)
        augmented_image[:, t_start:t_start + t_zero, i] = 0
    return augmented_image

# ======================================================
# 3. Dataset Loading (CORRECTED Data Generator)
# ======================================================

class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, image_cache_path, labels, indices, batch_size, mean, std, is_training=True, preload_chunks=True):
        self.labels = labels
        self.indices = indices
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.is_training = is_training
        self.preload_chunks = preload_chunks

        # Open memmap file
        self.image_data = np.memmap(image_cache_path, dtype=np.float32, mode='r', shape=(21799, 224, 224, 3))
        
        # --- preload subset of dataset into RAM if enabled ---
        if preload_chunks:
            subset_size = min(5000, len(indices))  # ~3 GB RAM for 5000 samples
            preload_idx = np.random.choice(indices, subset_size, replace=False)
            self.cache = {i: self.image_data[i] for i in preload_idx}
            print(f"🧠 Cached {subset_size} samples in RAM for fast access.")
        else:
            self.cache = {}
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X_batch, y_batch = [], []

        for idx in batch_indices:
            if idx in self.cache:
                img = self.cache[idx]
            else:
                img = self.image_data[idx]
            X_batch.append(img)
            y_batch.append(self.labels[idx])

        X_batch = np.array(X_batch, dtype=np.float32)
        y_batch = np.array(y_batch, dtype=np.float32)

        # Normalize
        X_batch = (X_batch - self.mean) / (self.std + 1e-7)

        # Augment
        if self.is_training:
            X_batch = np.array([spec_augment(img) for img in X_batch])
        
        return X_batch, y_batch


# ======================================================
# 4. Model & Training
# ======================================================
# --- Load metadata and define splits ---
Y = pd.read_csv(META_CSV_PATH, index_col="ecg_id")
y_labels = Y.age.values.astype(DTYPE)
train_idx = np.where((Y.strat_fold != TEST_FOLD) & (Y.strat_fold != VAL_FOLD))[0]
val_idx = np.where(Y.strat_fold == VAL_FOLD)[0]
test_idx = np.where(Y.strat_fold == TEST_FOLD)[0]

# --- Instantiate generators ---
train_generator = CustomDataGenerator(LOCAL_IMAGE_CACHE_PATH, y_labels, train_idx, BATCH_SIZE, channel_mean, channel_std)
# ======================================================
val_generator = CustomDataGenerator(LOCAL_IMAGE_CACHE_PATH, y_labels, val_idx, BATCH_SIZE, channel_mean, channel_std, is_training=False)
test_generator = CustomDataGenerator(LOCAL_IMAGE_CACHE_PATH, y_labels, test_idx, BATCH_SIZE, channel_mean, channel_std, is_training=False)

# --- Build and compile model (no changes needed here) ---
PRETRAINED_ECG_MODEL_PATH = os.path.join(DRIVE_BASE_PATH, "ecg_resnet_v3.keras")

print(f"🧠 Loading pre-trained ECG model from: {PRETRAINED_ECG_MODEL_PATH}")
# 1. Load your full pre-trained model
base_model = keras.models.load_model(PRETRAINED_ECG_MODEL_PATH)

# 2. Freeze the base model to protect its learned weights
base_model.trainable = True

# 3. Get the output of a layer BEFORE the incorrect final layers.
#    'dropout_6' is a good choice from your model summary.
# feature_extractor_output = base_model.get_layer('dropout_6').output

# # 4. Create a new regression "head" for our age prediction task
# x = keras.layers.Dense(256, activation="relu")(feature_extractor_output)
# x = keras.layers.Dropout(0.5)(x)
# # The final layer must output 1 value for the age
# age_prediction = keras.layers.Dense(1, name="age_prediction")(x)

# # 5. Create the new model by combining the base and the new head
# model = keras.Model(inputs=base_model.input, outputs=age_prediction)
model = base_model
# 6. Compile the new model
#    Use a standard learning rate since we are only training the new head.
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-6),
    loss="mean_squared_error",
    metrics=["mean_absolute_error"]
)

# print("✅ Model head replaced and re-compiled for regression.")
model.summary()


# --- Callbacks & Training ---
callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=BEST_MODEL_PATH, save_best_only=True, monitor="val_mean_absolute_error", mode="min"),
    keras.callbacks.EarlyStopping(monitor="val_mean_absolute_error", patience=8, mode="min", restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_mean_absolute_error", factor=0.2, patience=3, mode="min")
]

print("\n🔥 Starting model training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=callbacks
)

# --- Final Evaluation ---
print("\n🧪 Evaluating the best model on the test set...")
test_loss, test_mae = model.evaluate(test_generator)
print(f"\n🏆 Final Test MAE: {test_mae:.4f}")