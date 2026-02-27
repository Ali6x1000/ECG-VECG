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

DRIVE_BASE_PATH = "/content/drive/MyDrive/CNN"
META_CSV_PATH = os.path.join(DRIVE_BASE_PATH, "ptbxl_database.csv")
CHECKPOINT_DIR = os.path.join(DRIVE_BASE_PATH, "Best")
LOCAL_CACHE_DIR = "/content/cache"

ECG_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "resnet50_age_ECG_Only_best.pth")
VCG_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "resnet50_age_VCG_Only_best.pth")

ECG_CACHE_PATH = os.path.join(LOCAL_CACHE_DIR, "ECG_Only_age_uint8.dat")
VCG_CACHE_PATH = os.path.join(LOCAL_CACHE_DIR, "VCG_Only_age_uint8.dat")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
IMG_SIZE = (224, 224)
STORAGE_DTYPE = np.uint8
TRAIN_DTYPE = np.float32

print(f"Running on device: {DEVICE}")

# --- 2. HELPERS (Dataset & Feature Extractor) ---

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

class FeatureExtractor(nn.Module):
    """
    Wraps a ResNet and returns the 2048 features BEFORE the final linear layer.
    """
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        # Take everything EXCEPT the final 'fc' layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1) # Flatten (N, 2048, 1, 1) -> (N, 2048)

def load_feature_model(path):
    # 1. Load the original architecture
    base_model = models.resnet50(weights=None)
    num_ftrs = base_model.fc.in_features
    base_model.fc = nn.Linear(num_ftrs, 1)
    
    # 2. Load trained weights
    base_model.load_state_dict(torch.load(path, map_location=DEVICE))
    
    # 3. Strip the head
    extractor = FeatureExtractor(base_model)
    extractor.to(DEVICE)
    extractor.eval()
    return extractor

# --- 3. EXTRACTION ENGINE ---

def extract_deep_features(model_name, model, cache_path, df_fold, total_records):
    print(f"   ... Extracting features from {model_name}")
    
    dataset = RAMDataset(df_fold, cache_path, total_records)
    
    # ✅ FIX 1: Set num_workers=0 to prevent Multiprocessing crash
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=0, pin_memory=True)
    
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(DEVICE)
            
            # Get 2048-dim vector
            feats = model(images)
            
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
            
    # Concatenate all batches
    X = np.vstack(features_list)
    y = np.concatenate(labels_list)
    
    del loader, dataset
    gc.collect()
    
    return X, y

# --- 4. MAIN PIPELINE ---

def run_feature_fusion():
    print(f"\n{'='*40}")
    print(f"🧬 DEEP FEATURE FUSION (ResNet Features -> XGBoost)")
    print(f"{'='*40}")

    # 1. Load Metadata
    print("1. Loading Metadata...")
    df = pd.read_csv(META_CSV_PATH, index_col="ecg_id")
    df = df.dropna(subset=['filename_hr', 'age'])
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age'])
    df = df[(df['age'] >= 18) & (df['age'] <= 85)]
    df = df.reset_index(drop=False)
    df['memmap_idx'] = df.index.values
    total_records = len(df)

    # 2. Splits (Fold 9 Train, Fold 10 Test)
    val_df = df[df['strat_fold'] == 9].copy()
    test_df = df[df['strat_fold'] == 10].copy()

    # 3. Load Feature Extractors
    print("\n2. Loading Models as Feature Extractors...")
    ecg_extractor = load_feature_model(ECG_MODEL_PATH)
    vcg_extractor = load_feature_model(VCG_MODEL_PATH)

    # 4. Extract Features
    print("\n3. Extracting Deep Features (This may take a moment)...")
    
    # Train Set (Fold 9)
    print("   -> Processing Train Set (ECG)...")
    X_train_ecg, y_train = extract_deep_features("ECG", ecg_extractor, ECG_CACHE_PATH, val_df, total_records)
    print("   -> Processing Train Set (VCG)...")
    X_train_vcg, _       = extract_deep_features("VCG", vcg_extractor, VCG_CACHE_PATH, val_df, total_records)
    
    # Test Set (Fold 10)
    print("   -> Processing Test Set (ECG)...")
    X_test_ecg, y_test   = extract_deep_features("ECG", ecg_extractor, ECG_CACHE_PATH, test_df, total_records)
    print("   -> Processing Test Set (VCG)...")
    X_test_vcg, _        = extract_deep_features("VCG", vcg_extractor, VCG_CACHE_PATH, test_df, total_records)
    
    del ecg_extractor, vcg_extractor
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Concatenate Features
    print("\n4. Stacking Features...")
    X_train = np.hstack([X_train_ecg, X_train_vcg])
    X_test = np.hstack([X_test_ecg, X_test_vcg])
    
    print(f"   Train Input Shape: {X_train.shape} (Rows, Features)")
    print(f"   Test Input Shape:  {X_test.shape} (Rows, Features)")

    # 6. Train XGBoost
    print("\n5. Training XGBoost on Deep Features...")
    
    #  FIX 2: Moved early_stopping_rounds to the constructor
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,       
        learning_rate=0.01,     
        max_depth=4,            
        subsample=0.7,          
        colsample_bytree=0.5,   
        reg_alpha=0.1,          
        reg_lambda=0.1,         
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50 
    )
    
    
    xgb_reg.fit(X_train, y_train, 
                eval_set=[(X_test, y_test)], 
                verbose=50)
    
    # 7. Evaluate
    print("\n6. Final Evaluation...")
    preds = xgb_reg.predict(X_test)
    
    # --- Metrics ---
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)     # <--- ADDED MSE
    rmse = np.sqrt(mse)
    
    print(f"{'-'*30}")
    print(f"🏆 Deep Feature Fusion MAE:  {mae:.4f}")
    print(f"   Deep Feature Fusion MSE:  {mse:.4f}") # <--- PRINTED MSE
    print(f"   Deep Feature Fusion RMSE: {rmse:.4f}")
    print(f"{'-'*30}")

    # Save
    save_path = os.path.join(CHECKPOINT_DIR, "xgboost_deep_feature_fusion.json")
    xgb_reg.save_model(save_path)
    print(f"Saved model to {save_path}")

if __name__ == "__main__":
    run_feature_fusion()