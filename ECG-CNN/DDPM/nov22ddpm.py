import os
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from google.colab import drive
from shutil import copyfile
import pickle
import gc

# Mount drive
if not os.path.ismount('/content/drive'):
    drive.mount('/content/drive')

# ======================================================
# 0. Configuration & Housekeeping
# ======================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Clean up previous runs to free RAM
gc.collect()
torch.cuda.empty_cache()

TRAINING_MODE = 'ecg_only'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
AGE_VECTOR_DIM = 512
TIME_EMB_DIM = 256

# --- File Paths ---
DRIVE_BASE_PATH = "/content/drive/MyDrive/CNN"
META_CSV_PATH = os.path.join(DRIVE_BASE_PATH, "ptbxl_database.csv")
PICKLE_CACHE_PATH = os.path.join(DRIVE_BASE_PATH, "ecg_image_cache_hr_1x224x224.pkl")
STATS_PATH = os.path.join(DRIVE_BASE_PATH, f"ptbxl_{TRAINING_MODE}_stats.npz")
AGE_VECTOR_MODEL_PATH = os.path.join(DRIVE_BASE_PATH, f"age_vector_extractor_{TRAINING_MODE}.keras")
DIFFUSION_MODEL_WEIGHTS_PATH = os.path.join(DRIVE_BASE_PATH, f"diffusion_model_weights_{TRAINING_MODE}_v2.pth")

# [NEW] Cache paths to save progress on Drive
TRAIN_VECTORS_CACHE = os.path.join(DRIVE_BASE_PATH, f"cache_vectors_train_{TRAINING_MODE}.pt")
VAL_VECTORS_CACHE = os.path.join(DRIVE_BASE_PATH, f"cache_vectors_val_{TRAINING_MODE}.pt")

# Local paths
LOCAL_PICKLE_CACHE_PATH = "/content/source_image.pkl"
LOCAL_STATS_PATH = f"/content/ptbxl_{TRAINING_MODE}_stats.npz"
LOCAL_AGE_VECTOR_MODEL_PATH = f"/content/age_vector_extractor_{TRAINING_MODE}.keras"

# --- Copy Data ---
print("🚚 Copying data from Google Drive...")
if not os.path.exists(LOCAL_PICKLE_CACHE_PATH):
    copyfile(PICKLE_CACHE_PATH, LOCAL_PICKLE_CACHE_PATH)
if not os.path.exists(LOCAL_STATS_PATH) and os.path.exists(STATS_PATH):
    copyfile(STATS_PATH, LOCAL_STATS_PATH)
if not os.path.exists(LOCAL_AGE_VECTOR_MODEL_PATH):
    copyfile(AGE_VECTOR_MODEL_PATH, LOCAL_AGE_VECTOR_MODEL_PATH)
print("✅ Data check complete.")

# ======================================================
# 1. Load Data & Cache
# ======================================================
print("Loading data...")
Y_full = pd.read_csv(META_CSV_PATH, index_col="ecg_id")
N_RECORDS = len(Y_full)

# Load Keras age extractor
try:
    import keras
    age_vector_extractor = keras.models.load_model(LOCAL_AGE_VECTOR_MODEL_PATH)
    age_vector_extractor.trainable = False
except ImportError:
    print("⚠️ Keras not found. Ensure tensorflow is installed.")

# Load pickle data
with open(LOCAL_PICKLE_CACHE_PATH, 'rb') as f:
    pickle_data = pickle.load(f)

# Build Cache
print(f"🛠 Building {TRAINING_MODE} image cache...")
image_cache_in_memory = np.zeros((N_RECORDS, *IMG_SIZE, 3), dtype=np.float32)
valid_filenames = set(pickle_data.keys())

for i, (ecg_id, row) in enumerate(tqdm(Y_full.iterrows(), total=N_RECORDS, desc="Building cache")):
    filename = row["filename_hr"]
    if not pd.isna(filename) and filename in valid_filenames:
        ecg_img = pickle_data[filename].squeeze().astype(np.float32)
        image_cache_in_memory[i] = np.stack([ecg_img]*3, axis=-1)

del pickle_data

# Calculate/Load Stats
if not os.path.exists(LOCAL_STATS_PATH):
    print(f"🛠 Calculating stats...")
    pixel_sum = np.sum(image_cache_in_memory.astype(np.float64), axis=(0, 1, 2))
    pixel_sq_sum = np.sum(image_cache_in_memory.astype(np.float64)**2, axis=(0, 1, 2))
    num_pixels = N_RECORDS * IMG_SIZE[0] * IMG_SIZE[1]
    channel_mean = pixel_sum / num_pixels
    channel_std = np.sqrt(pixel_sq_sum / num_pixels - channel_mean**2)
    np.savez(LOCAL_STATS_PATH, mean=channel_mean, std=channel_std)
else:
    stats = np.load(LOCAL_STATS_PATH)
    channel_mean = stats['mean']
    channel_std = stats['std']

# ======================================================
# 2. Diffusion Dataset (With Drive Caching)
# ======================================================
class DiffusionDataset(Dataset):
    def __init__(self, image_cache, indices, age_vector_extractor, channel_mean, channel_std, cache_path):
        self.image_cache = image_cache
        self.indices = indices
        self.cache_path = cache_path
        self.age_vectors = self._get_age_vectors(age_vector_extractor, channel_mean, channel_std)
    
    def _get_age_vectors(self, model, mean, std):
        # 1. Check if cache exists on Drive
        if os.path.exists(self.cache_path):
            print(f"💾 Found cached vectors at {self.cache_path}. Loading...")
            return torch.load(self.cache_path)

        # 2. If not, compute them
        print(f"🧠 Computing vectors (Saving to {self.cache_path})...")
        all_vectors = []
        batch_size = 64
        
        mean_np = mean.reshape(1, 1, 3)
        std_np = std.reshape(1, 1, 3)
        
        for i in range(0, len(self.indices), batch_size):
            batch_idx = self.indices[i : i + batch_size]
            batch_images = self.image_cache[batch_idx]
            
            # Normalize
            batch_images = (batch_images - mean_np) / (std_np + 1e-7)
            
            # Predict
            vectors = model.predict_on_batch(batch_images)
            all_vectors.append(vectors)
            
            if i % (batch_size * 50) == 0 and i > 0:
                print(f"   Processed {i}/{len(self.indices)}...")

        final_vectors = np.concatenate(all_vectors, axis=0)
        final_tensor = torch.tensor(final_vectors, dtype=torch.float32)
        
        # 3. Save to Drive for next time
        torch.save(final_tensor, self.cache_path)
        print(f"✅ Saved vectors to Drive.")
        
        return final_tensor
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img = self.image_cache[self.indices[idx]]
        image = torch.tensor(img, dtype=torch.float32)
        image = (image / 127.5) - 1.0
        image = image.permute(2, 0, 1) # HWC -> CHW
        return {'image': image, 'age_vector': self.age_vectors[idx]}

# ======================================================
# 3. Model Components
# ======================================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(b, c, h, w)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, age_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.pool = nn.MaxPool2d(2)
        
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.age_proj = nn.Linear(age_emb_dim, out_channels)

    def forward(self, x, time_emb, age_vec):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        
        # Add Embeddings
        t_emb = self.time_proj(time_emb)[:, :, None, None]
        a_emb = self.age_proj(age_vec)[:, :, None, None]
        x = x + t_emb + a_emb
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        
        return x, self.pool(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, age_emb_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # --- FIX IS HERE ---
        # Input channels = in_channels (from below) + in_channels (from skip)
        self.conv1 = nn.Conv2d(in_channels * 2, out_channels, 3, padding=1)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.age_proj = nn.Linear(age_emb_dim, out_channels)

    def forward(self, x, skip_x, time_emb, age_vec):
        x = self.up(x)
        # Concatenate skip connection
        x = torch.cat([x, skip_x], dim=1)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        
        # Add Embeddings
        t_emb = self.time_proj(time_emb)[:, :, None, None]
        a_emb = self.age_proj(age_vec)[:, :, None, None]
        x = x + t_emb + a_emb
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        return x

class UNet(nn.Module):
    def __init__(self, age_emb_dim=512, time_emb_dim=256):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # Encoder
        self.down1 = DownBlock(3, 64, time_emb_dim, age_emb_dim)
        self.down2 = DownBlock(64, 128, time_emb_dim, age_emb_dim)
        self.down3 = DownBlock(128, 256, time_emb_dim, age_emb_dim)
        self.down4 = DownBlock(256, 512, time_emb_dim, age_emb_dim)
        
        # Bottleneck
        self.bot_conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.attn = SelfAttention(1024)
        self.bot_conv2 = nn.Conv2d(1024, 512, 3, padding=1)
        
        # Decoder
        self.up1 = UpBlock(512, 256, time_emb_dim, age_emb_dim)
        self.up2 = UpBlock(256, 128, time_emb_dim, age_emb_dim)
        self.up3 = UpBlock(128, 64, time_emb_dim, age_emb_dim)
        self.up4 = UpBlock(64, 64, time_emb_dim, age_emb_dim)
        
        self.out_conv = nn.Conv2d(64, 3, 1)

    def forward(self, x, t, age_vec):
        t_emb = self.time_mlp(t)
        
        x1, p1 = self.down1(x, t_emb, age_vec)
        x2, p2 = self.down2(p1, t_emb, age_vec)
        x3, p3 = self.down3(p2, t_emb, age_vec)
        x4, p4 = self.down4(p3, t_emb, age_vec)
        
        b = self.bot_conv1(p4)
        b = self.attn(b)
        b = self.bot_conv2(b)
        
        u1 = self.up1(b, x4, t_emb, age_vec)
        u2 = self.up2(u1, x3, t_emb, age_vec)
        u3 = self.up3(u2, x2, t_emb, age_vec)
        u4 = self.up4(u3, x1, t_emb, age_vec)
        
        return self.out_conv(u4)

# ======================================================
# 4. Diffusion Model & Schedule
# ======================================================
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

class DiffusionModel(nn.Module):
    def __init__(self, unet, diffusion_steps=1000):
        super().__init__()
        self.unet = unet
        self.diffusion_steps = diffusion_steps
        
        betas = cosine_beta_schedule(diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
    
    def forward(self, images, age_vectors, noise=None):
        batch_size = images.shape[0]
        timesteps = torch.randint(0, self.diffusion_steps, (batch_size,), device=images.device)
        
        if noise is None:
            noise = torch.randn_like(images)
            
        alpha_bars = self.alpha_bars[timesteps].view(-1, 1, 1, 1)
        noisy_images = torch.sqrt(alpha_bars) * images + torch.sqrt(1.0 - alpha_bars) * noise
        
        predicted_noise = self.unet(noisy_images, timesteps, age_vectors)
        return predicted_noise, noise

# ======================================================
# 5. Training Loop
# ======================================================
VAL_FOLD = 9
TEST_FOLD = 10
train_idx = np.where((Y_full.strat_fold != TEST_FOLD) & (Y_full.strat_fold != VAL_FOLD))[0]
val_idx = np.where(Y_full.strat_fold == VAL_FOLD)[0]

print("Creating datasets...")
# Pass cache paths here
train_dataset = DiffusionDataset(image_cache_in_memory, train_idx, age_vector_extractor, channel_mean, channel_std, TRAIN_VECTORS_CACHE)
val_dataset = DiffusionDataset(image_cache_in_memory, val_idx, age_vector_extractor, channel_mean, channel_std, VAL_VECTORS_CACHE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print("Building model...")
unet = UNet(age_emb_dim=AGE_VECTOR_DIM, time_emb_dim=TIME_EMB_DIM).to(device)
diffusion_model = DiffusionModel(unet).to(device)

optimizer = optim.AdamW(diffusion_model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.L1Loss()

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        age_vectors = batch['age_vector'].to(device)
        
        optimizer.zero_grad()
        predicted_noise, actual_noise = model(images, age_vectors)
        loss = criterion(predicted_noise, actual_noise)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images = batch['image'].to(device)
            age_vectors = batch['age_vector'].to(device)
            predicted_noise, actual_noise = model(images, age_vectors)
            loss = criterion(predicted_noise, actual_noise)
            total_loss += loss.item()
    return total_loss / len(loader)

# --- Start ---
print("\n🚀 Starting training...")
best_val_loss = float('inf')
patience = 20
patience_counter = 0

for epoch in range(150):
    print(f"\nEpoch {epoch + 1}/150")
    train_loss = train_epoch(diffusion_model, train_loader, optimizer, criterion, device)
    val_loss = validate(diffusion_model, val_loader, criterion, device)
    
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(diffusion_model.state_dict(), DIFFUSION_MODEL_WEIGHTS_PATH)
        print(f"✅ Saved best model (val_loss: {val_loss:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("Early stopping.")
        break