import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. CONFIGURATION ---

# Paths (Assumed based on previous scripts)
DRIVE_BASE_PATH = "/content/drive/MyDrive/CNN"
META_CSV_PATH = os.path.join(DRIVE_BASE_PATH, "ptbxl_database.csv")
CHECKPOINT_DIR = os.path.join(DRIVE_BASE_PATH, "Best")
LOCAL_CACHE_DIR = "/content/cache"

# Define the models to evaluate
# Format: "Name": ("Model_Path", "Cache_Path")
MODELS_CONFIG = {
    "Both": (
        os.path.join(CHECKPOINT_DIR, "resnet50_age_Both_best.pth"),
        os.path.join(LOCAL_CACHE_DIR, "Both_age_uint8.dat")
    ),
    "ECG_Only": (
        os.path.join(CHECKPOINT_DIR, "resnet50_age_ECG_Only_best.pth"),
        os.path.join(LOCAL_CACHE_DIR, "ECG_Only_age_uint8.dat")
    ),
    "VCG_Only": (
        os.path.join(CHECKPOINT_DIR, "resnet50_age_VCG_Only_best.pth"),
        os.path.join(LOCAL_CACHE_DIR, "VCG_Only_age_uint8.dat")
    )
}

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
IMG_SIZE = (224, 224)
STORAGE_DTYPE = np.uint8
TRAIN_DTYPE = np.float32

print(f"Running on device: {DEVICE}")
sns.set_style("whitegrid")

# --- 2. DATA CLASSES ---

class RAMDataset(Dataset):
    def __init__(self, df, memmap_path, total_records):
        self.indices = df['memmap_idx'].values
        self.labels = df['age'].values.astype(np.float32)
        
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Read-only memmap
        self.fp = np.memmap(memmap_path, dtype=STORAGE_DTYPE, mode='r', shape=(total_records, *IMG_SIZE, 3))
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        img = self.fp[global_idx].astype(TRAIN_DTYPE) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1)
        tensor = (tensor - self.imagenet_mean) / self.imagenet_std
        return tensor, torch.tensor(self.labels[idx], dtype=torch.float32)

def get_age_model():
    model = models.resnet50(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1) # Linear Head
    return model

# --- 3. INFERENCE ENGINE ---

def run_inference(model_name, model_path, cache_path, df_test, total_records):
    print(f"🔹 Running Inference: {model_name}...")
    
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found: {model_path}. Skipping.")
        return None
    if not os.path.exists(cache_path):
        print(f"⚠️ Cache not found: {cache_path}. Skipping.")
        return None

    dataset = RAMDataset(df_test, cache_path, total_records)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    model = get_age_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    preds = []
    
    with torch.no_grad():
        for images, _ in tqdm(loader, leave=False):
            images = images.to(DEVICE)
            outputs = model(images)
            preds.extend(outputs.cpu().numpy().flatten())
            
    del model, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    
    return np.array(preds)

# --- 4. PLOTTING HELPERS ---

def plot_scatter(ax, y_true, y_pred, name, color):
    # Calculate metrics for title
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    ax.scatter(y_true, y_pred, alpha=0.1, color=color, s=5)
    
    # Perfect fit line
    lims = [18, 85]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=10)
    
    ax.set_title(f"{name}\n$R^2$={r2:.3f} | MAE={mae:.2f}")
    ax.set_xlabel("True Age")
    ax.set_ylabel("Predicted Age")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

def plot_bland_altman(ax, y_true, y_pred, name, color):
    mean = (y_true + y_pred) / 2
    diff = y_pred - y_true
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    
    ax.scatter(mean, diff, alpha=0.1, color=color, s=5)
    ax.axhline(md, color='black', linestyle='-', lw=1)
    ax.axhline(md + 1.96*sd, color='red', linestyle='--', lw=1)
    ax.axhline(md - 1.96*sd, color='red', linestyle='--', lw=1)
    
    ax.set_title(f"Bland-Altman: {name}")
    ax.set_xlabel("Mean Age")
    ax.set_ylabel("Diff (Pred - True)")
    ax.set_ylim([-30, 30])

# --- 5. MAIN EXECUTION ---

def main():
    # A. Load Metadata
    print("📂 Loading Metadata...")
    df = pd.read_csv(META_CSV_PATH, index_col="ecg_id")
    df = df.dropna(subset=['filename_hr', 'age'])
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age'])
    df = df[(df['age'] >= 18) & (df['age'] <= 85)]
    
    # Reset index to match cache generation logic
    df = df.reset_index(drop=False)
    df['memmap_idx'] = df.index.values
    total_records = len(df)
    
    # Get Test Set (Fold 10)
    test_df = df[df['strat_fold'] == 10].copy()
    y_true = test_df['age'].values
    print(f"✅ Test Set Loaded: {len(test_df)} samples.")

    # B. Run Inference
    results = {}
    for name, (m_path, c_path) in MODELS_CONFIG.items():
        preds = run_inference(name, m_path, c_path, test_df, total_records)
        if preds is not None:
            results[name] = preds

    if not results:
        print("❌ No predictions generated. Check paths.")
        return

    # C. Statistical Table
    stats_data = []
    print("\n📊 --- STATISTICAL REPORT ---")
    for name, preds in results.items():
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        r, _ = stats.pearsonr(y_true, preds)
        rho, _ = stats.spearmanr(y_true, preds)
        r2 = r2_score(y_true, preds)
        
        stats_data.append({
            "Model": name,
            "MAE": mae, 
            "RMSE": rmse, 
            "Pearson r": r, 
            "Spearman rho": rho, 
            "R2": r2
        })
        
    df_stats = pd.DataFrame(stats_data).sort_values("MAE")
    print(df_stats.to_string(index=False))
    
    # D. P-Values (Comparing Both vs Others)
    if "Both" in results:
        print("\n🧪 --- P-VALUE COMPARISON (Paired T-Test) ---")
        err_both = np.abs(y_true - results["Both"])
        
        for name, preds in results.items():
            if name == "Both": continue
            err_other = np.abs(y_true - preds)
            t_stat, p_val = stats.ttest_rel(err_both, err_other)
            signif = "✅ Significant" if p_val < 0.05 else "❌ Not Significant"
            print(f"Both vs {name}: p={p_val:.2e} -> {signif}")

    # E. PLOTTING
    print("\n🎨 Generating Plots...")
    models_avail = list(results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
    
    # 1. Regression Scatter & Bland-Altman Grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for i, name in enumerate(models_avail):
        if i >= 3: break
        preds = results[name]
        c = colors[i]
        
        # Row 1: Scatter
        plot_scatter(axes[0, i], y_true, preds, name, c)
        
        # Row 2: Bland-Altman
        plot_bland_altman(axes[1, i], y_true, preds, name, c)
        
    plt.tight_layout()
    plt.show()
    
    # 2. Error Distribution (KDE)
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(models_avail):
        errors = results[name] - y_true
        sns.kdeplot(errors, fill=True, label=f"{name} (Mean: {np.mean(errors):.2f})", alpha=0.3, color=colors[i])
    plt.axvline(0, color='k', linestyle='--')
    plt.title("Error Distribution (Prediction - True)")
    plt.xlabel("Error (Years)")
    plt.legend()
    plt.show()
    
    # 3. Bias by Age Group (Regression to Mean Check)
    plt.figure(figsize=(12, 6))
    # Create bins
    test_df['Age_Group'] = pd.cut(test_df['age'], bins=[18, 30, 40, 50, 60, 70, 80, 90], 
                                  labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'])
    
    # Calculate MAE per group per model
    group_data = []
    for name in results:
        test_df[f'{name}_AbsErr'] = np.abs(test_df['age'] - results[name])
        
    # Melt for Seaborn
    melted = test_df.melt(id_vars=['Age_Group'], 
                          value_vars=[f'{name}_AbsErr' for name in results],
                          var_name='Model', value_name='MAE')
    
    melted['Model'] = melted['Model'].str.replace('_AbsErr', '')
    
    sns.boxplot(x='Age_Group', y='MAE', hue='Model', data=melted, palette=colors)
    plt.title("Mean Absolute Error by Age Group")
    plt.ylabel("Absolute Error (Years)")
    plt.xlabel("Age Decade")
    plt.show()

if __name__ == "__main__":
    main()

