import os
import pickle
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm.auto import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# --- 1. Configuration & Global Setup ---

# *** USER SETTING: Choose 'combined', 'ecg_only', or 'vcg_only' here ***
DATASET_TYPE = "combined"

WORLD_SIZE = torch.cuda.device_count()
USE_DDP = WORLD_SIZE > 1

# Define directories
RESULTS_DIR = "./results"
CHECKPOINT_DIR = "./checkpoints"
LOCAL_CACHE_DIR = "./cache"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

# HPC Paths
BATCH_DIR = "/home/aan90/ondemand/Ali/processed_batches"
HOSP_FOLDER = '/home/aan90/ondemand/Ali/physionet.org/files/mimiciv/3.1/hosp/'
RECORD_LIST = "/home/aan90/ondemand/Ali/record_list.csv"

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 64  # Good for P100/2080s
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4  # Added L2 regularization

# Early stopping
PATIENCE = 15
MIN_DELTA = 0.0  # require val_mae improvement of at least this amount

def setup_ddp():
    """Initialize distributed training ONCE"""
    if USE_DDP:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        # Increased timeout to 60 min to prevent crashes during data loading
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
        return local_rank
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0

def cleanup_ddp():
    if USE_DDP:
        dist.destroy_process_group()

# --- 2. Helper Functions ---

def load_mimic_metadata():
    print("Merging record list with patient age data...")
    df_rec = pd.read_csv(RECORD_LIST).dropna(subset=["path", "subject_id"])
    df_pat = pd.read_csv(os.path.join(HOSP_FOLDER, "patients.csv.gz")).dropna(subset=["anchor_age", "subject_id"])
    
    df = pd.merge(df_rec, df_pat, on="subject_id")
    df = df[(df["anchor_age"] >= 18) & (df["anchor_age"] <= 80)]
    
    # Stratified Split logic (approximate)
    df['fold'] = df['subject_id'] % 10
    
    df = df.reset_index(drop=True)
    df['memmap_idx'] = df.index
    print(f"✅ Metadata merged. {len(df)} records ready.")
    return df

def prepare_master_cache(dataset_type, df_meta):
    """Stitches batch_*.pkl files into one master binary file."""
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
                target_dict = batch_data.get(dataset_type, {})
                
                for path, img in target_dict.items():
                    if path in path_to_idx:
                        idx = path_to_idx[path]
                        if img.max() <= 1.05 and img.dtype != np.uint8:
                            img = img * 255.0
                        img_squeezed = img.squeeze()
                        img_3ch = np.stack([img_squeezed]*3, axis=-1).astype('uint8')
                        fp[idx] = img_3ch
        fp.flush()
        del fp
    return cache_path

def generate_loss_plot(history, model_name):
    plt.switch_backend('agg') 
    plt.figure(figsize=(10, 6))
    plt.plot(history['epochs'], history['train_mae'], label='Train MAE')
    plt.plot(history['epochs'], history['val_mae'], label='Validation MAE')
    plt.title(f'Training Progress: {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('MAE (Years)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_loss_graph.png"))
    plt.close()

# --- 3. Optimized Dataset with Augmentation ---

class MimicStreamDataset(Dataset):
    def __init__(self, df, shared_memmap, is_train=False):
        self.indices = df['memmap_idx'].values
        self.labels = df['anchor_age'].values.astype(np.float32)
        self.data = shared_memmap
        self.is_train = is_train

        # Base normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

        # Augmentation for Training Only
        if self.is_train:
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5), # Randomly flip signal
                transforms.RandomAffine(
                    degrees=5,             # Slight rotation
                    translate=(0.05, 0.05), # Slight shift
                    scale=(0.95, 1.05)      # Slight zoom
                ),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1) # Optional for greyscale
            ])
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Read from shared memory
        img = self.data[self.indices[idx]]

        # Convert to tensor (C, H, W) and float 0-1
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Apply Augmentation (Train only)
        if self.is_train:
            tensor = self.augment(tensor)
            
        # Apply Normalization
        tensor = self.normalize(tensor)
        
        return tensor, torch.tensor(self.labels[idx])

# --- 4. Training Engine ---

def train_model(name, memmap_path, df_meta, local_rank):
    is_main_process = local_rank == 0
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process:
        print(f"\n{'='*40}\nTraining Model: {name}\n{'='*40}")

    # Splits
    train_df = df_meta[df_meta['fold'] < 8]
    val_df = df_meta[df_meta['fold'] == 8]
    test_df = df_meta[df_meta['fold'] == 9]

    # --- SHARED MEMORY LOADING ---
    if is_main_process: print(f"⏳ Mapping {len(df_meta)} images into Shared Memory...")
    
    # Load ONCE into Page Cache (No private copy)
    shared_data = np.memmap(memmap_path, dtype='uint8', mode='r', 
                           shape=(len(df_meta), 224, 224, 3))
    
    if is_main_process: print("✅ Memory map created. Distributing to datasets...")

    # Pass shared data to datasets
    # Note: is_train=True enables augmentation for the training set
    train_ds = MimicStreamDataset(train_df, shared_data, is_train=True)
    val_ds = MimicStreamDataset(val_df, shared_data, is_train=False)
    test_ds = MimicStreamDataset(test_df, shared_data, is_train=False)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if USE_DDP else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if USE_DDP else None
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=4, pin_memory=True, 
        persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=val_sampler, 
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=4, pin_memory=True)

    # --- Model with Dropout ---
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Modify final layer to add Dropout for regularization
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),  # 50% Dropout
        nn.Linear(num_ftrs, 1)
    )
    
    model = model.to(device)
    if USE_DDP:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Added Weight Decay for Regularization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    best_val_mae = float('inf')
    epochs_no_improve = 0
    history = {'train_mae': [], 'val_mae': [], 'epochs': []}
    
    for epoch in range(NUM_EPOCHS):
        if USE_DDP: train_sampler.set_epoch(epoch)
        
        model.train()
        train_mae = 0.0
        
        if is_main_process:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        else:
            pbar = train_loader
            
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device).view(-1, 1)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                loss = criterion(preds, lbls)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_mae += torch.abs(preds - lbls).sum().item()
            
            if is_main_process and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({'mae': f"{train_mae / ((pbar.n + 1) * BATCH_SIZE):.2f}"})

        # Synchronize Metrics
        train_mae_tensor = torch.tensor(train_mae).to(device)
        if USE_DDP: dist.all_reduce(train_mae_tensor, op=dist.ReduceOp.SUM)
        
        total_train_samples = len(train_df)
        avg_train_mae = train_mae_tensor.item() / total_train_samples

        # Validation
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device).view(-1, 1)
                with torch.amp.autocast('cuda'):
                    preds = model(imgs)
                val_mae += torch.abs(preds - lbls).sum().item()
        
        val_mae_tensor = torch.tensor(val_mae).to(device)
        if USE_DDP: dist.all_reduce(val_mae_tensor, op=dist.ReduceOp.SUM)
        
        total_val_samples = len(val_df)
        avg_val_mae = val_mae_tensor.item() / total_val_samples
        
        if is_main_process:
            history['train_mae'].append(avg_train_mae)
            history['val_mae'].append(avg_val_mae)
            history['epochs'].append(epoch + 1)
            
            print(f"Epoch {epoch+1}: Train MAE {avg_train_mae:.2f} | Val MAE {avg_val_mae:.2f}")

            improved = (best_val_mae - avg_val_mae) > MIN_DELTA
            if improved:
                best_val_mae = avg_val_mae
                epochs_no_improve = 0
                save_obj = model.module.state_dict() if USE_DDP else model.state_dict()
                torch.save(save_obj, os.path.join(CHECKPOINT_DIR, f"{name}_best.pth"))
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1} (no val MAE improvement for {PATIENCE} epochs).")

        # Ensure all ranks stop together
        if USE_DDP:
            stop_tensor = torch.tensor(
                1 if (is_main_process and epochs_no_improve >= PATIENCE) else 0,
                device=device,
                dtype=torch.int
            )
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item() == 1:
                break
        else:
            if epochs_no_improve >= PATIENCE:
                break

    if is_main_process:
        print(f"Generating results for {name}...")
        generate_loss_plot(history, name)
        with open(os.path.join(RESULTS_DIR, f"{name}_history.json"), 'w') as f:
            json.dump(history, f)

    # Cleanup
    del model, optimizer, scaler, train_loader, val_loader, train_ds, val_ds, test_ds, shared_data
    torch.cuda.empty_cache()
    gc.collect()

# --- 5. Main Execution ---

if __name__ == "__main__":
    local_rank = setup_ddp()
    
    if local_rank == 0:
        df_meta = load_mimic_metadata()
    else:
        df_meta = None
    
    if USE_DDP:
        dist.barrier()
        df_meta = load_mimic_metadata()

    # --- SINGLE MODEL TRAINING ---
    if local_rank == 0:
        master_path = prepare_master_cache(DATASET_TYPE, df_meta)
    
    if USE_DDP:
        dist.barrier()
        master_path = os.path.join(LOCAL_CACHE_DIR, f"{DATASET_TYPE}_master.dat")

    train_model(DATASET_TYPE, master_path, df_meta, local_rank)
    
    cleanup_ddp()
    if local_rank == 0:
        print("All training complete.")