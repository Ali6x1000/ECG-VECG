import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm.auto import tqdm
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1. CONFIGURATION ---

# Paths
DRIVE_BASE_PATH = "/content/drive/MyDrive/CNN"
META_CSV_PATH = os.path.join(DRIVE_BASE_PATH, "ptbxl_database.csv")
CHECKPOINT_DIR = os.path.join(DRIVE_BASE_PATH, "Best")
LOCAL_CACHE_DIR = "/content/cache"

# Model Weights Paths (The .pth files you just trained)
ECG_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "resnet50_age_ECG_Only_best.pth")
VCG_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "resnet50_age_VCG_Only_best.pth")

# Cache Paths (The .dat files created in the previous step)
# Note: Ensure these match the filenames generated in your training script
ECG_CACHE_PATH = os.path.join(LOCAL_CACHE_DIR, "ECG_Only_age_uint8.dat")
VCG_CACHE_PATH = os.path.join(LOCAL_CACHE_DIR, "VCG_Only_age_uint8.dat")

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64  # Bigger batch size for inference is fine
IMG_SIZE = (224, 224)
STORAGE_DTYPE = np.uint8 # Matches our optimized storage
TRAIN_DTYPE = np.float32

print(f"Running on device: {DEVICE}")

# --- 2. SHARED CLASSES (MUST MATCH TRAINING SCRIPT) ---

class RAMDataset(Dataset):
    def __init__(self, df, memmap_path, total_records):
        self.indices = df['memmap_idx'].values
        self.labels = df['age'].values.astype(np.float32)
        
        # ImageNet Statistics
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Open Memmap in Read-Only mode
        self.fp = np.memmap(memmap_path, dtype=STORAGE_DTYPE, mode='r', shape=(total_records, *IMG_SIZE, 3))
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 1. Get global index
        global_idx = self.indices[idx]
        
        # 2. Read from disk/ram (uint8)
        img = self.fp[global_idx]
        
        # 3. Convert to float [0, 1]
        img = img.astype(TRAIN_DTYPE) / 255.0
        
        # 4. To Tensor & Normalize
        tensor = torch.from_numpy(img).permute(2, 0, 1)
        tensor = (tensor - self.imagenet_mean) / self.imagenet_std
        
        return tensor, torch.tensor(self.labels[idx], dtype=torch.float32)

def get_age_model():
    model = models.resnet50(weights=None) # Weights will be loaded from file
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1) # Linear Head (No ReLU)
    return model

# --- 3. INFERENCE ENGINE ---

def generate_predictions(model_name, model_path, cache_path, df_fold, total_records):
    """
    Runs inference on a dataset and returns predictions + actuals.
    """
    print(f"   ... Loading {model_name} from {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    # Setup Data
    dataset = RAMDataset(df_fold, cache_path, total_records)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # Setup Model
    model = get_age_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    preds = []
    actuals = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Predicting {model_name}"):
            images = images.to(DEVICE)
            outputs = model(images)
            preds.extend(outputs.cpu().numpy().flatten())
            actuals.extend(labels.numpy().flatten())
            
    # Cleanup
    del model, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    
    return np.array(preds), np.array(actuals)

# --- 4. MAIN PIPELINE ---

def run_stacking_pipeline():
    print(f"\n{'='*40}")
    print(f"🚀 STARTING XGBOOST STACKING PIPELINE")
    print(f"{'='*40}")

    # --- A. Load & Filter Metadata (MUST MATCH TRAINING EXACTLY) ---
    print("1. Loading and Filtering Metadata...")
    df = pd.read_csv(META_CSV_PATH, index_col="ecg_id")
    
    # Apply exact same filters as training
    df = df.dropna(subset=['filename_hr', 'age'])
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age'])
    df = df[(df['age'] >= 18) & (df['age'] <= 85)] # Age Filter
    
    # Re-establish Global Index
    df = df.reset_index(drop=False)
    df['memmap_idx'] = df.index.values
    
    total_records = len(df)
    print(f"   Total valid records: {total_records}")

    # --- B. Create Splits ---
    # We use Fold 9 (Validation) to TRAIN the XGBoost
    # We use Fold 10 (Test) to TEST the XGBoost
    val_df = df[df['strat_fold'] == 9].copy()
    test_df = df[df['strat_fold'] == 10].copy()
    
    print(f"   Training XGBoost on Fold 9 (n={len(val_df)})")
    print(f"   Testing XGBoost on Fold 10 (n={len(test_df)})")

    # --- C. Generate Predictions (The Stacking Features) ---
    
    # 1. Get ECG Predictions
    print("\n2. Generating ECG Predictions...")
    val_ecg_pred, val_y = generate_predictions("ECG_Model", ECG_MODEL_PATH, ECG_CACHE_PATH, val_df, total_records)
    test_ecg_pred, test_y = generate_predictions("ECG_Model", ECG_MODEL_PATH, ECG_CACHE_PATH, test_df, total_records)
    
    # 2. Get VCG Predictions
    print("\n3. Generating VCG Predictions...")
    val_vcg_pred, _ = generate_predictions("VCG_Model", VCG_MODEL_PATH, VCG_CACHE_PATH, val_df, total_records)
    test_vcg_pred, _ = generate_predictions("VCG_Model", VCG_MODEL_PATH, VCG_CACHE_PATH, test_df, total_records)
    
    # --- D. Build XGBoost Datasets ---
    print("\n4. formatting Data for XGBoost...")
    
    # Stack inputs horizontally: [[ECG_Pred_1, VCG_Pred_1], [ECG_Pred_2, VCG_Pred_2]...]
    X_train = np.column_stack((val_ecg_pred, val_vcg_pred))
    y_train = val_y
    
    X_test = np.column_stack((test_ecg_pred, test_vcg_pred))
    y_test = test_y
    
    print(f"   Train Input Shape: {X_train.shape}")
    print(f"   Test Input Shape:  {X_test.shape}")

    # --- E. Train XGBoost ---
    print("\n5. Training XGBoost...")
    
    # XGBoost Regressor
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=1.0,
        n_jobs=-1,
        random_state=42
    )
    
    xgb_reg.fit(X_train, y_train)
    
    # --- F. Evaluate Results ---
    print("\n6. Final Evaluation (Test Set Fold 10)...")
    
    # Individual Performance
    mae_ecg = mean_absolute_error(y_test, test_ecg_pred)
    mae_vcg = mean_absolute_error(y_test, test_vcg_pred)
    
    # Ensemble Performance
    stack_preds = xgb_reg.predict(X_test)
    mae_stack = mean_absolute_error(y_test, stack_preds)
    mse_stack = mean_squared_error(y_test, stack_preds)
    
    print(f"{'-'*30}")
    print(f"Single Model - ECG Only MAE: {mae_ecg:.4f}")
    print(f"Single Model - VCG Only MAE: {mae_vcg:.4f}")
    print(f"{'-'*30}")
    print(f"🏆 XGBoost Ensemble MAE:     {mae_stack:.4f}")
    print(f"   XGBoost Ensemble RMSE:    {np.sqrt(mse_stack):.4f}")
    print(f"{'-'*30}")
    
    # Save XGBoost Model
    save_path = os.path.join(CHECKPOINT_DIR, "xgboost_stacking_model.json")
    xgb_reg.save_model(save_path)
    print(f"✅ XGBoost model saved to: {save_path}")

    # Feature Importance (Which doctor did XGBoost listen to more?)
    print("\nFeature Importance:")
    print(f"   ECG Weight: {xgb_reg.feature_importances_[0]:.4f}")
    print(f"   VCG Weight: {xgb_reg.feature_importances_[1]:.4f}")

if __name__ == "__main__":
    try:
        run_stacking_pipeline()
    except Exception as e:
        print(f"Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()