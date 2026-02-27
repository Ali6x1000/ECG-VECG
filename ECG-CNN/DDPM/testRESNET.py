import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
import keras
from tqdm import tqdm
import pickle
from google.colab import drive
drive.mount('/content/drive')
# ==========================================
# 1. Configuration
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_path = "/content/drive/MyDrive/CNN/diffusion_model_weights_ecg_only_v2.pth"
keras_model_path = "/content/drive/MyDrive/CNN/age_vector_extractor_ecg_only.keras"
csv_path = "/content/drive/MyDrive/CNN/ptbxl_database.csv"
pickle_path = "/content/drive/MyDrive/CNN/source_image.pkl"

# How much noise to add? (0 = No change, 1000 = Total destruction)
# 500 is the "Sweet Spot" for changing style (age) but keeping identity.
START_STEP = 500

# ==========================================
# 2. Load Data & Models
# ==========================================
print("Loading Dataframe...")
Y_full = pd.read_csv(csv_path, index_col="ecg_id")

print("Loading Keras Age Extractor...")
age_extractor = keras.models.load_model(keras_model_path)
age_extractor.trainable = False

print("Loading Images...")
with open(pickle_path, 'rb') as f:
    pickle_data = pickle.load(f)

# ==========================================
# 3. Define Architecture
# ==========================================


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
        self.ff_self = nn.Sequential(nn.LayerNorm([channels]), nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))
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
        self.norm1 = nn.GroupNorm(8, out_channels); self.norm2 = nn.GroupNorm(8, out_channels); self.act = nn.SiLU(); self.pool = nn.MaxPool2d(2)
        self.time_proj = nn.Linear(time_emb_dim, out_channels); self.age_proj = nn.Linear(age_emb_dim, out_channels)
    def forward(self, x, time_emb, age_vec):
        x = self.conv1(x); x = self.norm1(x); x = self.act(x)
        t = self.time_proj(time_emb)[:, :, None, None]; a = self.age_proj(age_vec)[:, :, None, None]
        x = x + t + a
        x = self.conv2(x); x = self.norm2(x); x = self.act(x)
        return x, self.pool(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, age_emb_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels * 2, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels); self.norm2 = nn.GroupNorm(8, out_channels); self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_emb_dim, out_channels); self.age_proj = nn.Linear(age_emb_dim, out_channels)
    def forward(self, x, skip_x, time_emb, age_vec):
        x = self.up(x); x = torch.cat([x, skip_x], dim=1)
        x = self.conv1(x); x = self.norm1(x); x = self.act(x)
        t = self.time_proj(time_emb)[:, :, None, None]; a = self.age_proj(age_vec)[:, :, None, None]
        x = x + t + a
        x = self.conv2(x); x = self.norm2(x); x = self.act(x)
        return x

class UNet(nn.Module):
    def __init__(self, age_emb_dim=512, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU())
        self.down1 = DownBlock(3, 64, time_emb_dim, age_emb_dim); self.down2 = DownBlock(64, 128, time_emb_dim, age_emb_dim)
        self.down3 = DownBlock(128, 256, time_emb_dim, age_emb_dim); self.down4 = DownBlock(256, 512, time_emb_dim, age_emb_dim)
        self.bot_conv1 = nn.Conv2d(512, 1024, 3, padding=1); self.attn = SelfAttention(1024); self.bot_conv2 = nn.Conv2d(1024, 512, 3, padding=1)
        self.up1 = UpBlock(512, 256, time_emb_dim, age_emb_dim); self.up2 = UpBlock(256, 128, time_emb_dim, age_emb_dim)
        self.up3 = UpBlock(128, 64, time_emb_dim, age_emb_dim); self.up4 = UpBlock(64, 64, time_emb_dim, age_emb_dim)
        self.out_conv = nn.Conv2d(64, 3, 1)
    def forward(self, x, t, age_vec):
        t_emb = self.time_mlp(t)
        x1, p1 = self.down1(x, t_emb, age_vec); x2, p2 = self.down2(p1, t_emb, age_vec)
        x3, p3 = self.down3(p2, t_emb, age_vec); x4, p4 = self.down4(p3, t_emb, age_vec)
        b = self.bot_conv1(p4); b = self.attn(b); b = self.bot_conv2(b)
        u1 = self.up1(b, x4, t_emb, age_vec); u2 = self.up2(u1, x3, t_emb, age_vec)
        u3 = self.up3(u2, x2, t_emb, age_vec); u4 = self.up4(u3, x1, t_emb, age_vec)
        return self.out_conv(u4)

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

print("Loading Diffusion Weights...")
model = DiffusionModel(UNet(age_emb_dim=512, time_emb_dim=256)).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()
print("✅ Models Loaded.")
# ==========================================
# 4. Helpers: Robust Vector Extraction
# ==========================================

import torch.nn.functional as F
from scipy import signal as scipy_signal
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def calculate_metrics(original, reconstructed):
    # 1. Convert Tensors to Numpy if necessary
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    # 2. Extract 1D Signal from Image if passed as (H, W, C) or (C, H, W)
    # Assumption: If input is huge (e.g. > 1000 items), it's likely an image we need to collapse
    if original.ndim > 1:
        # If it's (224, 224, 3), take the mean of the first channel (Red) to get 1D signal
        if original.shape[-1] == 3:
             original = original[:, :, 0].mean(axis=0)
        # If it's (1, 3, 224, 224)
        elif original.ndim == 4:
             original = original[0, 0, :, :].mean(axis=0)
        # If it's just (N, 1), flatten it
        else:
             original = original.flatten()

    if reconstructed.ndim > 1:
        if reconstructed.shape[-1] == 3:
             reconstructed = reconstructed[:, :, 0].mean(axis=0)
        elif reconstructed.ndim == 4:
             reconstructed = reconstructed[0, 0, :, :].mean(axis=0)
        else:
             reconstructed = reconstructed.flatten()

    # 3. Final Safety Check: Strictly 1D
    original = original.flatten()
    reconstructed = reconstructed.flatten()

    # --- METRICS CALCULATION ---

    # 1. MSE (Mean Squared Error)
    mse = np.mean((original - reconstructed) ** 2)

    # 2. DTW (Dynamic Time Warping) - Distance
    # This measures similarity even if the signal is shifted in time
    distance, path = fastdtw(original, reconstructed, dist=euclidean)

    # 3. Pearson Correlation (Shape similarity)
    correlation = np.corrcoef(original, reconstructed)[0, 1]

    return {
        "MSE": mse,
        "DTW_Distance": distance,
        "Correlation": correlation
    }


# ===== Helper Functions =====
def estimate_heart_rate(signal, sampling_rate=500):
    """Estimate heart rate (BPM) from 1D signal."""
    from scipy.signal import find_peaks
    # Ensure signal is 1D numpy
    signal = signal.flatten()

    # Find peaks (R-peaks)
    peaks, _ = find_peaks(signal, height=0.4, distance=15)

    if len(peaks) < 2:
        return 0.0

    # Calculate mean RR interval
    rr_intervals = np.diff(peaks) / sampling_rate  # seconds
    mean_rr = np.mean(rr_intervals)

    # Convert to BPM
    hr = 60.0 / mean_rr if mean_rr > 0 else 0.0
    return hr

def estimate_qrs_width(signal, sampling_rate=500):
    """Estimate QRS width (ms)."""
    from scipy.signal import find_peaks, peak_widths
    signal = signal.flatten()

    # Find R-peaks
    peaks, properties = find_peaks(signal, height=0.4, distance=15)

    if len(peaks) == 0:
        return 0.0

    # Calculate widths at half height
    results_half = peak_widths(signal, peaks, rel_height=0.5)
    widths_samples = results_half[0]

    # Convert samples to ms (samples / rate * 1000)
    widths_ms = (widths_samples / sampling_rate) * 1000
    return np.mean(widths_ms) if len(widths_ms) > 0 else 0.0

def calculate_spectral_similarity(signal1, signal2):
    """Compare frequency content."""
    from scipy.signal import welch
    signal1 = signal1.flatten()
    signal2 = signal2.flatten()

    # Compute power spectral density
    f1, psd1 = welch(signal1, fs=100, nperseg=min(len(signal1), 64))
    f2, psd2 = welch(signal2, fs=100, nperseg=min(len(signal2), 64))

    # Normalize
    psd1 = psd1 / (np.sum(psd1) + 1e-10)
    psd2 = psd2 / (np.sum(psd2) + 1e-10)

    # Correlation
    corr, _ = pearsonr(psd1, psd2)
    return corr

# --- Main Metrics Function ---

def calculate_metrics(original, reconstructed):
    """
    Calculates comprehensive metrics.
    Handles Tensor/Numpy conversion and Shape flattening automatically.
    """
    # 1. Convert Tensors to Numpy
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    # 2. Un-Normalize: [-1, 1] -> [0, 1] for SSIM/PSNR
    original_norm = (original + 1) / 2
    reconstructed_norm = (reconstructed + 1) / 2
    original_norm = np.clip(original_norm, 0, 1)
    reconstructed_norm = np.clip(reconstructed_norm, 0, 1)

    # 3. Extract 1D Signals for Physics Metrics (Mean of Red Channel)
    if original_norm.ndim > 1:
        if original_norm.shape[-1] == 3: # (H, W, 3)
             sig1 = original_norm[:, :, 0].mean(axis=0)
             sig2 = reconstructed_norm[:, :, 0].mean(axis=0)
        elif original_norm.ndim == 4: # (1, 3, H, W)
             sig1 = original_norm[0, 0, :, :].mean(axis=0)
             sig2 = reconstructed_norm[0, 0, :, :].mean(axis=0)
        else: # (H, W)
             sig1 = original_norm.mean(axis=0)
             sig2 = reconstructed_norm.mean(axis=0)
    else:
        sig1 = original_norm
        sig2 = reconstructed_norm

    # 4. Flatten for 1D math (Fixes ValueError)
    sig1_flat = sig1.flatten()
    sig2_flat = sig2.flatten()

    # --- CALCULATIONS ---

    # Basic Error
    mse = np.mean((original_norm - reconstructed_norm) ** 2)
    l1 = np.mean(np.abs(original_norm - reconstructed_norm))

    # Image Quality (PSNR)
    try:
        psnr_val = psnr(original_norm, reconstructed_norm, data_range=1.0)
    except:
        psnr_val = 0.0

    # SSIM (Requires 2D+)
    try:
        # Squeeze to remove batch dims for SSIM
        orig_sq = original_norm.squeeze()
        recon_sq = reconstructed_norm.squeeze()
        if orig_sq.ndim >= 2:
            ssim_val = ssim(orig_sq, recon_sq, data_range=1.0, channel_axis=-1 if orig_sq.ndim==3 else None)
        else:
            ssim_val = 0.0
    except:
        ssim_val = 0.0

    # Signal Metrics (DTW, Correlation, Spectral)
    try:
        distance, _ = fastdtw(sig1_flat, sig2_flat, dist=euclidean)
    except:
        distance = -1.0

    try:
        correlation = np.corrcoef(sig1_flat, sig2_flat)[0, 1]
    except:
        correlation = 0.0

    try:
        spec_sim = calculate_spectral_similarity(sig1_flat, sig2_flat)
    except:
        spec_sim = 0.0

    # Physics Metrics (HR, QRS)
    hr_orig = estimate_heart_rate(sig1_flat)
    hr_recon = estimate_heart_rate(sig2_flat)
    qrs_orig = estimate_qrs_width(sig1_flat)
    qrs_recon = estimate_qrs_width(sig2_flat)

    # --- RETURN DICTIONARY (Crucial Fix) ---
    return {
        "MSE": mse,
        "L1": l1,
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "DTW_Distance": distance,
        "Correlation": correlation,
        "Spectral_Similarity": spec_sim,
        "Original_HR": hr_orig,
        "Reconstructed_HR": hr_recon,
        "Original_QRS_Width": qrs_orig,
        "Reconstructed_QRS_Width": qrs_recon
    }



# 1. LOAD REAL STATS
stats_path = "/content/ptbxl_ecg_only_stats.npz"

if os.path.exists(stats_path):
    stats = np.load(stats_path)
    REAL_MEAN = stats['mean']
    REAL_STD = stats['std']
    print(f"✅ Loaded Real Stats: Mean={REAL_MEAN.mean():.4f}")
else:
    print("⚠️ Stats file not found! Using fallback arrays.")
    # FIX: Create actual NumPy arrays so .reshape() works
    REAL_MEAN = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    REAL_STD = np.array([1.0, 1.0, 1.0], dtype=np.float32)

def get_vector_for_age(target_age):
    """
    Robustly finds a vector.
    1. Searches wide window (Age +/- 5 years)
    2. Uses REAL normalization stats
    """
    # Step 1: Find a patient in a wider range
    window = 0
    candidates = pd.DataFrame()

    while len(candidates) == 0 and window < 10:
        candidates = Y_full[(Y_full['age'] >= target_age - window) & (Y_full['age'] <= target_age + window)]
        window += 1

    if len(candidates) == 0:
        print(f"❌ CRITICAL: No patient found even close to Age {target_age}!")
        return torch.zeros(1, 512).to(device)

    # Step 2: Pick the first valid file
    for ecg_id, row in candidates.iterrows():
        fname = row['filename_hr']
        if fname in pickle_data:
            # Get Raw Image
            raw = pickle_data[fname].squeeze().astype(np.float32)
            img_stack = np.stack([raw]*3, axis=-1)

            # Step 3: CORRECT NORMALIZATION
            # Ensure input is at least 1D before reshaping
            if isinstance(REAL_MEAN, np.ndarray):
                mean_poly = REAL_MEAN.reshape(1, 1, 3)
                std_poly = REAL_STD.reshape(1, 1, 3)
            else:
                # Safety net if logic fails
                mean_poly = 0.0
                std_poly = 1.0

            img_norm = (img_stack - mean_poly) / (std_poly + 1e-7)
            img_norm = np.expand_dims(img_norm, 0)

            # Step 4: Extract Vector
            vec = age_extractor.predict(img_norm, verbose=0)

            found_age = row['age']
            print(f"   ✅ Age {target_age}: Using Patient {ecg_id} (Real Age: {found_age})")

            return torch.tensor(vec, dtype=torch.float32).to(device)

    print(f"⚠️ candidates found for {target_age} but no pickle files.")
    return torch.zeros(1, 512).to(device)

# Use this updated function in your existing loop:
# images, titles = run_aging(patient_id, target_ages=[25, 50, 75, 90])

def get_specific_patient_image(ecg_id):
    fname = Y_full.loc[ecg_id]['filename_hr']
    raw = pickle_data[fname].squeeze().astype(np.float32)
    img_stack = np.stack([raw]*3, axis=-1)

    # Normalize for Diffusion [-1, 1]
    # --- THE FIX IS HERE ---
    tensor = torch.tensor(img_stack, dtype=torch.float32).permute(2, 0, 1)
    tensor = (tensor / 127.5) - 1.0
    return tensor.unsqueeze(0).to(device)

# ==========================================
# 5. The Core Function: Image-to-Image Aging
# ==========================================
@torch.no_grad()
def run_aging(start_ecg_id, target_ages=[20, 50, 80]):

    # 1. Load Original Patient
    original_age = Y_full.loc[start_ecg_id]['age']
    x_original = get_specific_patient_image(start_ecg_id)

    print(f"\n🧬 processing Patient {start_ecg_id} (Original Age: {original_age})")
    print(f"Adding noise up to step {START_STEP}...")

    # 2. Forward Diffuse to t
    noise = torch.randn_like(x_original)
    t_start = torch.tensor([START_STEP], device=device)

    alpha_bar = model.alpha_bars[t_start].view(-1, 1, 1, 1)
    x_noisy = torch.sqrt(alpha_bar) * x_original + torch.sqrt(1 - alpha_bar) * noise

    results = [x_original.cpu()]
    labels = [f"Original (Age {original_age})"]

    # 3. Denoise with different Age Vectors
    for age in target_ages:
        print(f"   ✨ Aging to {age}...")
        age_vec = get_vector_for_age(age)

        img = x_noisy.clone()

        for t in tqdm(reversed(range(START_STEP)), desc=f"Age {age}", leave=False):
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)

            predicted_noise = model.unet(img, t_tensor, age_vec)

            alpha = model.alphas[t]
            alpha_bar = model.alpha_bars[t]
            beta = model.betas[t]

            if t > 0:
                z = torch.randn_like(img)
            else:
                z = torch.zeros_like(img)

            coeff1 = 1 / torch.sqrt(alpha)
            coeff2 = (1 - alpha) / (torch.sqrt(1 - alpha_bar))
            mean = coeff1 * (img - coeff2 * predicted_noise)
            sigma = torch.sqrt(beta)

            img = mean + sigma * z

        results.append(img.cpu())
        labels.append(f"Modified (Age {age})")

    return results, labels

# ==========================================
# 6. Execution
# ==========================================

# Image Metrics (on 2D data if available, else 1D)
    # Note: SSIM needs 2D or 3D. If inputs are already flat, skip SSIM.
    if original_norm.ndim >= 2:
        # SSIM requires data_range specification
        # Squeeze to remove batch dim if present (1, H, W, C) -> (H, W, C)
        orig_sq = original_norm.squeeze()
        recon_sq = reconstructed_norm.squeeze()

        # Handle Channel dimension for SSIM
        multichannel = True if orig_sq.ndim == 3 else False
        win_size = min(7, orig_sq.shape[0], orig_sq.shape[1]) # Dynamic window size

        try:
            ssim_val = ssim(orig_sq, recon_sq, data_range=1.0, channel_axis=-1 if multichannel else None, win_size=win_size)
        except ValueError:
            ssim_val = 0.0 # Fallback if image too small
    else:
        ssim_val = 0.0

    psnr_val = psnr(original_norm, reconstructed_norm, data_range=1.0)
    mse = np.mean((original_norm - reconstructed_norm) ** 2)
    l1_loss = np.mean(np.abs(original_norm - reconstructed_norm))

    # Signal Metrics (1D)
    distance, _ = fastdtw(sig1_flat, sig2_flat, dist=euclidean)
    correlation = np.corrcoef(sig1_flat, sig2_flat)[0, 1]
    spec_sim = calculate_spectral_similarity(sig1_flat, sig2_flat)

    # Physics Metrics
    hr_orig = estimate_heart_rate(sig1_flat)
    hr_recon = estimate_heart_rate(sig2_flat)
    qrs_orig = estimate_qrs_width(sig1_flat)
    qrs_recon = estimate_qrs_width(sig2_flat)

    return {
        "MSE": mse,
        "L1": l1_loss,
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "DTW_Distance": distance,
        "Correlation": correlation,
        "Spectral_Similarity": spec_sim,
        "Original_HR": hr_orig,
        "Reconstructed_HR": hr_recon,
        "Original_QRS_Width": qrs_orig,
        "Reconstructed_QRS_Width": qrs_recon
    }
# Pick a patient
patient_id = Y_full.index[1]
target = [20,30,50,60,70]
# Run Aging Generation
images, titles = run_aging(patient_id, target_ages= target)

print("\n📊 Evaluation Metrics:")
print("=" * 80)

# Extract original signal for comparison
original_tensor = images[0]
all_metrics = []

# Loop starting from index 1 (the generated ages), comparing against index 0 (original)
for i in range(1, len(images)):
    target_age = target[i-1]

    # Calculate metrics between Original and Generated
    metrics = calculate_metrics(original_tensor, images[i])

    print(f"\n🔍 Original vs. Age {target_age}:")
    print(f"   PSNR:           {metrics['PSNR']} dB")
    print(f"   SSIM:           {metrics['SSIM']}")
    print(f"   Correlation:    {metrics['Correlation']}")
    print(f"   MSE:            {metrics['MSE']}")
    print(f"   HR Original:    {metrics['Original_HR']} BPM")
    print(f"   HR Modified:    {metrics['Reconstructed_HR']} BPM")
    print(f"   QRS Original:   {metrics['Original_QRS_Width']} ms")
    print(f"   QRS Modified:   {metrics['Reconstructed_QRS_Width']} ms")

    all_metrics.append((target_age, metrics))

print("\n" + "=" * 80)

# --- PLOTTING ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Quantitative Evaluation Metrics', fontsize=16, fontweight='bold')

ages = [m[0] for m in all_metrics]

# Plot 1: PSNR & SSIM
ax = axes[0, 0]
psnr_vals = [m[1]['PSNR'] for m in all_metrics]
ssim_vals = [m[1]['SSIM'] for m in all_metrics]
ax.plot(ages, psnr_vals, 'o-', label='PSNR', linewidth=2, color='blue')
ax.set_ylabel('PSNR (dB)', color='blue')
ax.tick_params(axis='y', labelcolor='blue')
ax2 = ax.twinx()
ax2.plot(ages, ssim_vals, 's-', color='red', label='SSIM', linewidth=2)
ax2.set_ylabel('SSIM', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax.set_xlabel('Target Age')
ax.set_title('Image Quality')
ax.grid(True, alpha=0.3)

# Plot 2: Signal Correlation
ax = axes[0, 1]
corr_vals = [m[1]['Correlation'] for m in all_metrics]
ax.plot(ages, corr_vals, 'o-', linewidth=2, markersize=8, color='green')
ax.set_xlabel('Target Age')
ax.set_ylabel('Pearson Correlation')
ax.set_title('Signal Similarity')
ax.grid(True, alpha=0.3)

# Plot 3: Heart Rate
ax = axes[0, 2]
hr_orig = [m[1]['Original_HR'] for m in all_metrics]
hr_recon = [m[1]['Reconstructed_HR'] for m in all_metrics]
ax.plot(ages, hr_orig, 'o-', label='Original', linewidth=2, color='gray')
ax.plot(ages, hr_recon, 's-', label='Modified', linewidth=2, color='orange')
ax.set_title('Heart Rate (BPM)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: QRS Width
ax = axes[1, 0]
qrs_orig = [m[1]['Original_QRS_Width'] for m in all_metrics]
qrs_recon = [m[1]['Reconstructed_QRS_Width'] for m in all_metrics]
ax.plot(ages, qrs_orig, 'o-', label='Original', linewidth=2, color='gray')
ax.plot(ages, qrs_recon, 's-', label='Modified', linewidth=2, color='purple')
ax.set_title('QRS Width (ms)')
ax.set_xlabel('Target Age')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: MSE & L1
ax = axes[1, 1]
mse_vals = [m[1]['MSE'] for m in all_metrics]
l1_vals = [m[1]['L1'] for m in all_metrics]
ax.plot(ages, mse_vals, 'o-', label='MSE', linewidth=2)
ax.plot(ages, l1_vals, 's-', label='L1', linewidth=2)
ax.set_title('Reconstruction Error')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Plot 6: Spectral Similarity
ax = axes[1, 2]
spec_vals = [m[1]['Spectral_Similarity'] for m in all_metrics]
ax.plot(ages, spec_vals, 'o-', linewidth=2, markersize=8, color='magenta')
ax.set_title('Frequency Domain Sim.')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/CNN/evaluation.png", dpi=300)
plt.show()

# --- Plot the Images ---
plt.figure(figsize=(20, 8))
for i, img_tensor in enumerate(images):
    img = (img_tensor.squeeze().permute(1, 2, 0).cpu().numpy() + 1) / 2
    img = np.clip(img, 0, 1)

    plt.subplot(1, len(images), i+1)
    plt.title(titles[i], fontsize=12, fontweight='bold')
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig("/content/drive/MyDrive/CNN/generated_ecg_images.png", dpi=300)
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# Overlay Plot Configuration
# ==========================================
# distinct colors for the progression:
# Black (Original) -> Blue (Young) -> ... -> Red (Old)
colors = ['black', 'blue', 'green', 'orange', 'red']
linestyles = ['-', '--', '-', '-', '-']
linewidths = [2.0, 1.5, 1.5, 1.5, 1.5]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})

# --- 1. Main Overlay Loop ---
signals = [] # Store for difference calculation

for i, img_tensor in enumerate(images):
    # Process Image Tensor -> 1D Signal
    img = (img_tensor.squeeze().permute(1, 2, 0).cpu().numpy() + 1) / 2
    img = np.clip(img, 0, 1)

    # Extract 1D signal (Average of Red Channel vertical pixels)
    signal = img[:, :, 0].mean(axis=0)
    signals.append(signal)

    # Plot on Top Axis
    # Use modulo % in case you generate more images than colors defined
    c = colors[i % len(colors)]
    ls = linestyles[i % len(linestyles)]
    lw = linewidths[i % len(linewidths)]

    ax1.plot(signal, label=titles[i], color=c, linestyle=ls, linewidth=lw, alpha=0.7)

# Formatting Top Plot
ax1.set_title(f"ECG Aging Progression Overlay (Patient {patient_id})", fontsize=14, fontweight='bold')
ax1.set_ylabel("Normalized Voltage", fontsize=12)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, which='both', alpha=0.3)
ax1.set_xlim(0, 224)

# --- 2. Difference Map (Oldest - Youngest) ---
# This shows ONLY the features that changed
if len(signals) > 1:
    # Assuming index 1 is youngest generated and index -1 is oldest generated
    # (Adjust indices if your 'target_ages' list order is different)
    youngest_sig = signals[1]
    oldest_sig = signals[-1]

    diff = oldest_sig - youngest_sig

    ax2.plot(diff, color='purple', linewidth=1.5, label='Difference (Old - Young)')
    ax2.fill_between(range(len(diff)), diff, 0, where=(diff>0), color='red', alpha=0.3, label='Voltage Increase')
    ax2.fill_between(range(len(diff)), diff, 0, where=(diff<0), color='blue', alpha=0.3, label='Voltage Decrease')

    ax2.set_title("Difference Map: What exactly changed?", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time (Samples)", fontsize=12)
    ax2.set_ylabel("Delta", fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 224)

plt.tight_layout()

# --- Save & Show ---
save_path = "/content/drive/MyDrive/CNN/ecg_overlay_comparison.png"
plt.savefig(save_path, dpi=300)
print(f"✅ Plot saved to {save_path}")
plt.show()