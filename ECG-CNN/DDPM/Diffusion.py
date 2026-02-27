import os
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from tqdm import tqdm
from google.colab import drive
from shutil import copyfile
import pickle

drive.mount('/content/drive')

# --- FIX 1: Restore Mixed Precision & set correct DTYPE ---
# This is critical for memory efficiency.
# Using float16 cuts your data memory usage in half.




os.environ["KERAS_BACKEND"] = "tensorflow"
# ======================================================
# 0. Configuration
# ======================================================

# --- ⌘  CHOOSE YOUR EXPERIMENT (Must match Project 1) ---
TRAINING_MODE = 'ecg_only'
# ---

IMG_SIZE = (224, 224)
BATCH_SIZE = 16 # <-- NOTE: Reduced batch size. Diffusion is memory-heavy.
DTYPE = np.float32 # <-- FIX 2: Use float16 to save RAM
AGE_VECTOR_DIM = 512 # Must match the "age_vector_layer" in Project 1

# --- File Paths ---
DRIVE_BASE_PATH = "/content/drive/MyDrive/CNN"
META_CSV_PATH = os.path.join(DRIVE_BASE_PATH, "ptbxl_database.csv")

# --- MODIFIED: Path to the SOURCE Pickle file ---
PICKLE_CACHE_PATH = os.path.join(DRIVE_BASE_PATH, "ecg_image_cache_hr_1x224x224.pkl")

# Paths from Project 1
STATS_PATH = os.path.join(DRIVE_BASE_PATH, f"ptbxl_{TRAINING_MODE}_stats.npz")
AGE_VECTOR_MODEL_PATH = os.path.join(DRIVE_BASE_PATH, f"age_vector_extractor_{TRAINING_MODE}.keras") # From Step 1

# Paths for this project
DIFFUSION_MODEL_PATH = os.path.join(DRIVE_BASE_PATH, f"diffusion_model_{TRAINING_MODE}.keras")
DIFFUSION_MODEL_WEIGHTS_PATH = os.path.join(DRIVE_BASE_PATH, f"diffusion_model_weights_{TRAINING_MODE}.weights.h5")


# --- Local Paths (for Colab speed) ---
LOCAL_PICKLE_CACHE_PATH = "/content/source_image.pkl" # NEW
LOCAL_STATS_PATH = f"/content/ptbxl_{TRAINING_MODE}_stats.npz"
LOCAL_AGE_VECTOR_MODEL_PATH = f"/content/age_vector_extractor_{TRAINING_MODE}.keras"

# --- Copying data ---
print("🚚 Copying data from Google Drive...")
try:
    # --- MODIFIED: Copy pickle file ---
    copyfile(PICKLE_CACHE_PATH, LOCAL_PICKLE_CACHE_PATH)
    copyfile(STATS_PATH, LOCAL_STATS_PATH)
    copyfile(AGE_VECTOR_MODEL_PATH, LOCAL_AGE_VECTOR_MODEL_PATH)
    print("✅ Data copied locally.")
except FileNotFoundError as e:
    if str(e).find('_stats.npz') != -1:
        print("⚠️ Stats file not found. Will generate it.")
    else:
        print(f"❌ Error copying files: {e}. Check Drive paths.")
        raise
except Exception as e:
    print(f"❌ Error copying files: {e}. Check Drive paths.")
    raise


# ======================================================
# 1. Load Data & Models
# ======================================================
print("Loading Project 1 artifacts...")
try:
    Y_full = pd.read_csv(META_CSV_PATH, index_col="ecg_id")
    N_RECORDS = len(Y_full)

    # Load the Age Expert!
    age_vector_extractor = keras.models.load_model(LOCAL_AGE_VECTOR_MODEL_PATH)
    age_vector_extractor.trainable = False # FREEZE IT

    # --- MODIFIED: Load pickle data ---
    print(f"Loading pickle file from {LOCAL_PICKLE_CACHE_PATH}...")
    with open(LOCAL_PICKLE_CACHE_PATH, 'rb') as f:
        pickle_data = pickle.load(f)
    print(f"✅ Loaded pickle file with {len(pickle_data)} entries.")

except Exception as e:
    print(f"❌ Error loading artifacts: {e}")
    raise

# ======================================================
# 1.5 NEW: Generate Image Cache IN-MEMORY from Pickle
# ======================================================
# This will consume ~6.1 GB of RAM (because DTYPE is float16)
print(f"🛠  Building {TRAINING_MODE} image cache in-memory from pickle...")
try:
    # Create a large array in RAM
    image_cache_in_memory = np.zeros((N_RECORDS, *IMG_SIZE, 3), dtype=DTYPE)

    valid_filenames = set(pickle_data.keys())
    missing_count = 0

    for i, (ecg_id, row) in enumerate(tqdm(Y_full.iterrows(), total=N_RECORDS, desc=f"Building {TRAINING_MODE} cache")):
        filename = row["filename_hr"]

        if pd.isna(filename) or filename not in valid_filenames:
            missing_count += 1
            continue # Leave as zeros

        # Load images, squeeze (1, 224, 224) -> (224, 224)
        ecg_img = pickle_data[filename].squeeze().astype(DTYPE)
        # vcg_img = pickle_data[filename]['vcg'].squeeze().astype(DTYPE)

        final_image = np.zeros((*IMG_SIZE, 3), dtype=DTYPE)

        if TRAINING_MODE == 'ecg_only':
            final_image[:, :, 0] = ecg_img
            final_image[:, :, 1] = ecg_img
            final_image[:, :, 2] = ecg_img

        image_cache_in_memory[i] = final_image

    del pickle_data # Free up RAM
    print(f"✅ In-memory cache built ({image_cache_in_memory.nbytes / 1e9:.2f} GB).")
    if missing_count > 0:
        print(f"⚠️ Warning: {missing_count} records from metadata had no matching pickle entry.")

except Exception as e:
    print(f"❌ Error building in-memory cache: {e}")
    raise


# ======================================================
# 1.6 NEW: Calculate/Load Normalization Stats
# ======================================================
if not os.path.exists(LOCAL_STATS_PATH):
    print(f"🛠  Global stats for {TRAINING_MODE} not found. Calculating from in-memory cache...")

    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    num_pixels = N_RECORDS * IMG_SIZE[0] * IMG_SIZE[1]

    pixel_sum = np.sum(image_cache_in_memory.astype(np.float64), axis=(0, 1, 2))
    pixel_sq_sum = np.sum(image_cache_in_memory.astype(np.float64)**2, axis=(0, 1, 2))

    channel_mean = pixel_sum / num_pixels
    channel_std = np.sqrt(pixel_sq_sum / num_pixels - channel_mean**2)

    np.savez(LOCAL_STATS_PATH, mean=channel_mean, std=channel_std)
    copyfile(LOCAL_STATS_PATH, STATS_PATH)
    print(f"✅ Stats calculated and saved for {TRAINING_MODE}.")
else:
    stats = np.load(LOCAL_STATS_PATH)
    channel_mean = stats['mean']
    channel_std = stats['std']
    print(f"✅ Per-channel stats for {TRAINING_MODE} loaded.")

print(f"Channel Mean: {channel_mean}")
print(f"Channel Std: {channel_std}")


# ======================================================
# 2. Custom Data Generator for Diffusion
# ======================================================

# --- FIX 3: Make the Generator "Lazy" (Memory Efficient) ---
# This generator no longer pre-calculates everything.
# It processes ONE BATCH at a time in __getitem__.

class DiffusionDataGenerator(keras.utils.Sequence):
    def __init__(self, image_cache_in_memory, indices, age_vector_extractor,
                 diff_normalizer, resnet_normalizer, batch_size):

        print("🧠 Generator Initialized (Lazy Mode)")
        self.indices = indices
        self.image_cache = image_cache_in_memory
        self.age_vector_extractor = age_vector_extractor

        # We need two different normalizers
        self.diff_normalizer = diff_normalizer     # For diffusion model (to [-1, 1])
        self.resnet_normalizer = resnet_normalizer # For ResNet (age expert)

        self.batch_size = batch_size

        # --- NO PRE-CALCULATION HERE ---
        # This saves all the RAM.

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        # 1. Get the indices for this batch
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        # 2. Get the raw images for *this batch only*
        # This is a TINY copy (e.g., 16 images), not 21,000
        raw_images_batch = self.image_cache[batch_indices]

        # 3. Process images for the Diffusion Model (Input)
        images_normalized = self.diff_normalizer(raw_images_batch)

        # 4. Process images for the ResNet (Conditioning)
        normalized_images_for_resnet = self.resnet_normalizer(raw_images_batch)

        # 5. Calculate age vectors for *this batch only*
        # This is very fast and uses almost no RAM
        age_vectors_batch = self.age_vector_extractor.predict_on_batch(
            normalized_images_for_resnet
        )

        return {
            "image": images_normalized,
            "age_vector": age_vectors_batch.astype(DTYPE) # Ensure it's float16
        }

# ======================================================
# 3. Build the Diffusion Model
# ======================================================
# (This section is unchanged, but it will now correctly use DTYPE=float16)

# --- Helper functions for the U-Net ---
def DownBlock(width, block_depth, embedding_dim=32):
    def apply(x_and_t):
        x, t = x_and_t
        for _ in range(block_depth):
            x = keras.layers.Conv2D(width, kernel_size=3, padding="same")(x)
            t_proj = keras.layers.Dense(width)(t)
            t_proj = keras.layers.Reshape((1, 1, width))(t_proj)
            x = x + t_proj
            x = keras.layers.GroupNormalization(groups=8)(x)
            x = keras.layers.Activation("swish")(x)
        x = keras.layers.AveragePooling2D(pool_size=2)(x)
        return x
    return apply

def UpBlock(width, block_depth, embedding_dim=32):
    def apply(x_and_t):
        x, t = x_and_t
        x = keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = keras.layers.Conv2D(width, kernel_size=3, padding="same")(x)
            t_proj = keras.layers.Dense(width)(t)
            t_proj = keras.layers.Reshape((1, 1, width))(t_proj)
            x = x + t_proj
            x = keras.layers.GroupNormalization(groups=8)(x)
            x = keras.layers.Activation("swish")(x)
        return x
    return apply

def build_unet(img_size, embedding_dim):
    image_input = keras.Input(shape=(*img_size, 3), name="image", dtype=DTYPE)
    age_vector_input = keras.Input(shape=(embedding_dim,), name="age_vector", dtype=DTYPE)
    x = DownBlock(32, block_depth=2, embedding_dim=embedding_dim)([image_input, age_vector_input])
    x = DownBlock(64, block_depth=2, embedding_dim=embedding_dim)([x, age_vector_input])
    x = DownBlock(96, block_depth=2, embedding_dim=embedding_dim)([x, age_vector_input])
    x = DownBlock(128, block_depth=2, embedding_dim=embedding_dim)([x, age_vector_input])
    x = UpBlock(128, block_depth=2, embedding_dim=embedding_dim)([x, age_vector_input])
    x = UpBlock(96, block_depth=2, embedding_dim=embedding_dim)([x, age_vector_input])
    x = UpBlock(64, block_depth=2, embedding_dim=embedding_dim)([x, age_vector_input])
    x = UpBlock(32, block_depth=2, embedding_dim=embedding_dim)([x, age_vector_input])
    # Output in float32 for stability, as required by mixed precision
    output = keras.layers.Conv2D(3, kernel_size=1, padding="same", dtype=np.float32)(x)
    return keras.Model(inputs=[image_input, age_vector_input], outputs=output, name="conditional_unet")

# --- The Keras DiffusionModel Class ---
class DiffusionModel(keras.Model):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.mae_tracker = keras.metrics.Mean(name="mae_loss")
        self.diffusion_steps = 1000
        betas = tf.linspace(1e-4, 0.02, self.diffusion_steps)
        self.alphas = 1.0 - betas
        self.alpha_bars = tf.math.cumprod(self.alphas)

    def compile(self, optimizer):
        super().compile(optimizer=optimizer)

# Add this new method to your DiffusionModel class
    def test_step(self, data):
        images = data["image"]
        age_vectors = data["age_vector"]
        timesteps = tf.random.uniform(
            shape=(tf.shape(images)[0],), minval=0, maxval=self.diffusion_steps, dtype=tf.int32
        )
        noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
        alpha_bars = tf.gather(self.alpha_bars, timesteps)
        alpha_bars = tf.reshape(alpha_bars, (-1, 1, 1, 1))
        noisy_images = tf.sqrt(alpha_bars) * images + tf.sqrt(1.0 - alpha_bars) * noise

        # No GradientTape or optimizer steps.
        # Just get the prediction. Set training=False.
        predicted_noise = self.unet([noisy_images, age_vectors], training=False)
        loss = keras.losses.mean_absolute_error(noise, predicted_noise)
        loss = tf.reduce_mean(loss)

        # Update the metric and return
        self.mae_tracker.update_state(loss)
        return {"loss": self.mae_tracker.result()}
    # --- FIX 4: Restore the CORRECT train_step for Mixed Precision ---
    def train_step(self, data):
        images = data["image"]
        age_vectors = data["age_vector"]
        timesteps = tf.random.uniform(
            shape=(tf.shape(images)[0],), minval=0, maxval=self.diffusion_steps, dtype=tf.int32
        )
        noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
        alpha_bars = tf.gather(self.alpha_bars, timesteps)
        alpha_bars = tf.reshape(alpha_bars, (-1, 1, 1, 1))
        noisy_images = tf.sqrt(alpha_bars) * images + tf.sqrt(1.0 - alpha_bars) * noise

        with tf.GradientTape() as tape:
            predicted_noise = self.unet([noisy_images, age_vectors], training=True)
            loss = keras.losses.mean_absolute_error(noise, predicted_noise)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.unet.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.unet.trainable_variables))

        self.mae_tracker.update_state(loss)
        return {"loss": self.mae_tracker.result()}

    @property
    def metrics(self):
        return [self.mae_tracker]

# ======================================================
# 4. Training
# ======================================================
print("\n--- Initializing Model and Data ---")

# --- Normalization setup ---
# We now need TWO normalizers
# 1. For Diffusion model (images to [-1, 1])
diff_normalizer = lambda x: (x.astype(np.float32) / 127.5) - 1.0
# 2. For ResNet model (images to (x - mean) / std)
resnet_normalizer = lambda x: (x.astype(np.float32) - channel_mean) / (channel_std + 1e-7)


# --- Get data splits (same as Project 1) ---
VAL_FOLD = 9
TEST_FOLD = 10
train_idx = np.where((Y_full.strat_fold != TEST_FOLD) & (Y_full.strat_fold != VAL_FOLD))[0]
val_idx = np.where(Y_full.strat_fold == VAL_FOLD)[0]

# --- Create generators ---
# --- MODIFIED: Pass both normalizers ---
print("Initializing Train Generator (Lazy Mode)...")
train_generator = DiffusionDataGenerator(
    image_cache_in_memory, # <-- Pass the main array
    train_idx,
    age_vector_extractor,
    diff_normalizer,
    resnet_normalizer,
    BATCH_SIZE
)
print("\nInitializing Validation Generator (Lazy Mode)...")
val_generator = DiffusionDataGenerator(
    image_cache_in_memory, # <-- Pass the main array
    val_idx,
    age_vector_extractor,
    diff_normalizer,
    resnet_normalizer,
    BATCH_SIZE
)
print("✅ Data generators are ready.")

# --- Build and compile the model ---
unet = build_unet(IMG_SIZE, AGE_VECTOR_DIM)
# unet.summary()

diffusion_model = DiffusionModel(unet)

diffusion_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4) # Start with 1e-4 or 1e-5
)

print("🛠  Model built and compiled. Starting training...")

# --- Train the model ---
history = diffusion_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=150, # Train for more epochs
    callbacks=[
        keras.callbacks.ModelCheckpoint(DIFFUSION_MODEL_WEIGHTS_PATH, save_best_only=True, monitor="val_loss", mode="min", save_weights_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, mode="min")
    ]
)

print(f"✅ Training complete. Model weights saved to {DIFFUSION_MODEL_WEIGHTS_PATH}")

# Clean up
del image_cache_in_memory # <-- Free the 6.1 GB of RAM
del train_generator
del val_generator
del age_vector_extractor

print("\n--- 🎉 Project 2 Finished ---")
print("You are now ready for Project 3 (Interactive Inference).")