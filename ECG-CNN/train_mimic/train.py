import os
import pickle
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm.auto import tqdm

# --- 1. Configuration & Paths ---

# Update these to your HPC paths
BATCH_DIR = "/home/aan90/ondemand/Ali/processed_batches"
HOSP_FOLDER = '/home/aan90/ondemand/Ali/physionet.org/files/mimiciv/3.1/hosp/'
RECORD_LIST = "/home/aan90/ondemand/Ali/record_list.csv"
CHECKPOINT_DIR = "./checkpoints"
LOCAL_CACHE_DIR = "./cache"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

# Training Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 64  # Increased for HPC GPUs
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Optimized Metadata Loading ---

def load_mimic_metadata():
    print("Merging record list with patient age data...")
    df_rec = pd.read_csv(RECORD_LIST).dropna(subset=["path", "subject_id"])
    df_pat = pd.read_csv(os.path.join(HOSP_FOLDER, "patients.csv.gz")).dropna(subset=["anchor_age", "subject_id"])
    
    df = pd.merge(df_rec, df_pat, on="subject_id")
    df = df[(df["anchor_age"] >= 18) & (df["anchor_age"] <= 80)]
    
    # Stratify by subject_id to ensure no patient overlap between train/val/test
    # We use a simple hash of the subject_id for reproducible splits
    df['fold'] = df['subject_id'] % 10
    
    df = df.reset_index(drop=True)
    df['memmap_idx'] = df.index
    print(f"✅ Metadata merged. {len(df)} records ready.")
    return df

# --- 3. Master Cache Stitching ---

def prepare_master_cache(dataset_type, df_meta):
    """
    Stitches batch_*.pkl files into one master binary file for on-demand loading.
    dataset_type: 'combined', 'ecg_only', or 'vcg_only'
    """
    cache_path = os.path.join(LOCAL_CACHE_DIR, f"{dataset_type}_master.dat")
    n_records = len(df_meta)
    
    if not os.path.exists(cache_path):
        print(f"🛠️ Creating master cache for {dataset_type}...")
        fp = np.memmap(cache_path, dtype='uint8', mode='w+', shape=(n_records, 224, 224, 3))
        
        path_to_idx = {row['path']: i for i, row in df_meta.iterrows()}
        batch_files = [f for f in os.listdir(BATCH_DIR) if f.endswith('.pkl')]
        
        for b_file in tqdm(batch_files, desc="Stitching Batches"):
            with open(os.path.join(BATCH_DIR, b_file), 'rb') as f:
                batch_data = pickle.load(f)
                # Look for the specific key in your pickle dictionary
                target_dict = batch_data.get(dataset_type, {})
                
                for path, img in target_dict.items():
                    if path in path_to_idx:
                        idx = path_to_idx[path]
                        # Reshape (1, 224, 224) to (224, 224, 3) uint8
                        img_3ch = np.stack([img.squeeze()]*3, axis=-1).astype('uint8')
                        fp[idx] = img_3ch
        fp.flush()
        del fp
    return cache_path

# --- 4. Streamed Dataset Class (64GB RAM Friendly) ---

class MimicStreamDataset(Dataset):
    def __init__(self, df, memmap_path, total_len):
        self.indices = df['memmap_idx'].values
        self.labels = df['anchor_age'].values.astype(np.float32)
        self.memmap_path = memmap_path
        self.total_len = total_len
        
        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Memory-map access is extremely fast and uses almost no RAM
        fp = np.memmap(self.memmap_path, dtype='uint8', mode='r', shape=(self.total_len, 224, 224, 3))
        
        img = fp[self.indices[idx]].astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        
        return tensor, torch.tensor(self.labels[idx])

# --- 5. Training Logic ---

def train_model(name, memmap_path, df_meta):
    print(f"\nTraining Model: {name}")
    
    Patience = 10 

    # Splits based on folds (80% train, 10% val, 10% test)
    train_df = df_meta[df_meta['fold'] < 8]
    val_df = df_meta[df_meta['fold'] == 8]
    test_df = df_meta[df_meta['fold'] == 9]

    train_ds = MimicStreamDataset(train_df, memmap_path, len(df_meta))
    val_ds = MimicStreamDataset(val_df, memmap_path, len(df_meta))

    # Optimization: High num_workers and prefetch_factor to saturate GPU
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=8, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=8, pin_memory=True)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    best_mae = float('inf')
    patience_counter = 0 

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_mae = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).view(-1, 1)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                loss = criterion(preds, lbls)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_mae += torch.abs(preds - lbls).sum().item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).view(-1, 1)
                preds = model(imgs)
                val_mae += torch.abs(preds - lbls).sum().item()
        
        avg_val_mae = val_mae / len(val_ds)
        print(f"Epoch {epoch+1} Results - Train MAE: {train_mae/len(train_ds):.2f}, Val MAE: {avg_val_mae:.2f}")

        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            patience_counter = 0  # Reset counter
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{name}_best.pth"))
            print(f"✅ New best model saved (MAE: {best_mae:.2f})")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement for {patience_counter} epochs")
            
            if patience_counter >= Patience:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break  # Exit training loop
# --- 6. Main Execution ---

if __name__ == "__main__":
    df_meta = load_mimic_metadata()
    
    # You can loop through your different generated sets
    for dataset_type in ['combined', 'ecg_only', 'vcg_only']:
        # Ensure your batch files actually contain these keys
        master_path = prepare_master_cache(dataset_type, df_meta)
        train_model(dataset_type, master_path, df_meta)