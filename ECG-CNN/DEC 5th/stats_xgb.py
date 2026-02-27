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
import xgboost as xgb

# --- 1. CONFIGURATION ---

DRIVE_BASE_PATH = "/content/drive/MyDrive/CNN"
META_CSV_PATH = os.path.join(DRIVE_BASE_PATH, "ptbxl_database.csv")
CHECKPOINT_DIR = os.path.join(DRIVE_BASE_PATH, "Best")
LOCAL_CACHE_DIR = "/content/cache"

# ✅ Using the 'stacking' model (Scalar inputs)
XGB_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "xgboost_stacking_model.json")

# ResNet Model Paths
RESNET_CONFIG = {
    "ECG_Only": (
        os.path.join(CHECKPOINT_DIR, "resnet50_age_ECG_Only_best.pth"),
        os.path.join(LOCAL_CACHE_DIR, "ECG_Only_age_uint8.dat")
    ),
    "VCG_Only": (
        os.path.join(CHECKPOINT_DIR, "resnet50_age_VCG_Only_best.pth"),
        os.path.join(LOCAL_CACHE_DIR, "VCG_Only_age_uint8.dat")
    ),
    "Both_ResNet": (
        os.path.join(CHECKPOINT_DIR, "resnet50_age_Both_best.pth"),
        os.path.join(LOCAL_CACHE_DIR, "Both_age_uint8.dat")
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
    model.fc = nn.Linear(num_ftrs, 1) 
    return model

# --- 3. INFERENCE ENGINE (RESNET) ---

def run_resnet_inference(model_name, model_path, cache_path, df_test, total_records):
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

# --- 4. MAIN EXECUTION ---

def main():
    # A. Load Metadata
    print("📂 Loading Metadata...")
    df = pd.read_csv(META_CSV_PATH, index_col="ecg_id")
    df = df.dropna(subset=['filename_hr', 'age'])
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age'])
    df = df[(df['age'] >= 18) & (df['age'] <= 85)]
    
    df = df.reset_index(drop=False)
    df['memmap_idx'] = df.index.values
    total_records = len(df)
    
    # Get Test Set (Fold 10)
    test_df = df[df['strat_fold'] == 10].copy()
    y_true = test_df['age'].values
    print(f"✅ Test Set Loaded: {len(test_df)} samples.")

    # B. Generate ResNet Predictions
    results = {}
    
    for name, (m_path, c_path) in RESNET_CONFIG.items():
        preds = run_resnet_inference(name, m_path, c_path, test_df, total_records)
        if preds is not None:
            results[name] = preds

    # C. Run XGBoost Inference
    if "ECG_Only" in results and "VCG_Only" in results and os.path.exists(XGB_MODEL_PATH):
        print(f"🔹 Running Inference: XGBoost Ensemble...")
        X_test = np.column_stack((results["ECG_Only"], results["VCG_Only"]))
        
        try:
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(XGB_MODEL_PATH)
            xgb_preds = xgb_model.predict(X_test)
            results["Ensemble_XGB"] = xgb_preds
            print("✅ XGBoost predictions generated.")
        except Exception as e:
            print(f"⚠️ XGBoost Failed: {e}")
    else:
        print("⚠️ Skipping XGBoost: Missing inputs or model file.")

    if not results:
        print("❌ No predictions generated.")
        return

    # D. Statistical Table
    stats_data = []
    print("\n📊 --- FULL COMPARISON REPORT ---")
    for name, preds in results.items():
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        r, _ = stats.pearsonr(y_true, preds)
        rho, _ = stats.spearmanr(y_true, preds)
        r2 = r2_score(y_true, preds)
        
        stats_data.append({
            "Model": name,
            "MAE": mae, "RMSE": rmse, "Pearson r": r, "Spearman rho": rho, "R2": r2
        })
        
    df_stats = pd.DataFrame(stats_data).sort_values("MAE")
    print(df_stats.to_string(index=False))
    
    # E. P-Values
    if "Ensemble_XGB" in results:
        print("\n🧪 --- P-VALUE COMPARISON (XGBoost vs Others) ---")
        err_xgb = np.abs(y_true - results["Ensemble_XGB"])
        
        for name, preds in results.items():
            if name == "Ensemble_XGB": continue
            err_other = np.abs(y_true - preds)
            t_stat, p_val = stats.ttest_rel(err_xgb, err_other)
            signif = "✅ Significant" if p_val < 0.05 else "❌ Not Significant"
            better_worse = "Better" if t_stat < 0 else "Worse"
            print(f"XGB vs {name}: {better_worse} (p={p_val:.2e}) -> {signif}")

    # F. PLOTTING
    print("\n🎨 Generating Plots...")
    models_to_plot = list(results.keys())
    
    # 1. Regression Scatter & Bland-Altman
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'] 
    
    for i, name in enumerate(models_to_plot):
        if i >= 4: break
        preds = results[name]
        c = colors[i % len(colors)]
        
        # Row 1: Scatter
        r2 = r2_score(y_true, preds)
        mae = mean_absolute_error(y_true, preds)
        ax_scat = axes[0, i]
        ax_scat.scatter(y_true, preds, alpha=0.1, color=c, s=5)
        lims = [18, 85]
        ax_scat.plot(lims, lims, 'k--', alpha=0.75)
        ax_scat.set_title(f"{name}\nMAE={mae:.2f} | $R^2$={r2:.3f}")
        ax_scat.set_xlabel("True Age")
        ax_scat.set_ylabel("Predicted")
        ax_scat.set_xlim(lims)
        ax_scat.set_ylim(lims)
        
        # Row 2: Bland-Altman
        ax_ba = axes[1, i]
        mean = (y_true + preds) / 2
        diff = preds - y_true
        md = np.mean(diff)
        sd = np.std(diff)
        ax_ba.scatter(mean, diff, alpha=0.1, color=c, s=5)
        ax_ba.axhline(md, color='k', ls='-', label=f'Bias: {md:.2f}')
        ax_ba.axhline(md + 1.96*sd, color='r', ls='--')
        ax_ba.axhline(md - 1.96*sd, color='r', ls='--')
        ax_ba.set_title(f"Bland-Altman: {name}")
        ax_ba.set_xlabel("Mean Age")
        ax_ba.set_ylim([-30, 30])
        ax_ba.legend()
        
    plt.tight_layout()
    plt.show()

    # --- 🆕 2. ERROR DISTRIBUTION GRAPH (Restored) ---
    print("\nGenerating Error Distribution Graph...")
    plt.figure(figsize=(12, 6))
    
    for i, name in enumerate(models_to_plot):
        if i >= 4: break
        if name == "Ensemble_XGB" or name == "Both_ResNet":
            # Calculate Error
            error = results[name] - y_true
            mean_bias = np.mean(error)
        
            # Plot KDE
            sns.kdeplot(error, label=f"{name} (Bias: {mean_bias:.2f})", 
                        fill=True, alpha=0.2, color=colors[i % len(colors)], linewidth=2)
        
    plt.axvline(0, color='black', linestyle='--', alpha=0.8, label="Perfect Accuracy")
    plt.title("Error Distribution (Density Plot)")
    plt.xlabel("Error (Predicted Age - True Age)")
    plt.ylabel("Density")
    plt.xlim(-25, 25) # Focus on the center
    plt.legend()
    plt.show()

    # 3. Bias by Age Group (Safe Version)
    print("\nGenerating Age Group Analysis...")
    plt.figure(figsize=(14, 6))
    
    age_bins = [18, 30, 40, 50, 60, 70, 80, 90]
    age_labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    
    # Use raw numpy array for cutting to prevent index issues
    age_groups = pd.cut(y_true, bins=age_bins, labels=age_labels)
    
    melt_list = []
    for name in results:
        # Create Independent DataFrame
        df_model = pd.DataFrame({
            'Age_Group': age_groups,
            'MAE': np.abs(y_true - results[name]),
            'Model': name
        })
        melt_list.append(df_model)
    
    melted = pd.concat(melt_list, ignore_index=True)
    
    sns.boxplot(x='Age_Group', y='MAE', hue='Model', data=melted, palette=colors)
    plt.title("Mean Absolute Error by Age Group")
    plt.ylabel("Absolute Error (Years)")
    plt.show()

if __name__ == "__main__":
    main()