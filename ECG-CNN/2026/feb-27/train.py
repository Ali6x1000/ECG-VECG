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
from datetime import timedelta
import matplotlib.pyplot as plt

# --- 1. Configuration & Global Setup ---
# combined , ecg_only , vcg_only
DATASET_TYPE = "combined"

WORLD_SIZE = torch.cuda.device_count()
USE_DDP = WORLD_SIZE > 1

RESULTS_DIR = "./results"
CHECKPOINT_DIR = "./checkpoints/feb-27"
LOCAL_CACHE_DIR = "./cache"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

BATCH_DIR = "/home/aan90/ondemand/Ali/processed_batches"
HOSP_FOLDER = "/home/aan90/ondemand/Ali/physionet.org/files/mimiciv/3.1/hosp/"
RECORD_LIST = "/home/aan90/ondemand/Ali/record_list.csv"

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# --- DDP Setup ---

def setup_ddp():
    if USE_DDP:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=60)
        )
        return local_rank
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0

def cleanup_ddp():
    if USE_DDP:
        dist.destroy_process_group()

# --- 2. Metadata & Cache ---

def load_mimic_metadata():
    print("Merging record list with patient age data...")
    df_rec = pd.read_csv(RECORD_LIST).dropna(subset=["path", "subject_id"])
    df_pat = pd.read_csv(
        os.path.join(HOSP_FOLDER, "patients.csv.gz")
    ).dropna(subset=["anchor_age", "subject_id"])

    df = pd.merge(df_rec, df_pat, on="subject_id")
    df = df[(df.anchor_age >= 18) & (df.anchor_age <= 80)]

    # Stratified split using hash of subject_id to prevent leakage
    df["fold"] = df["subject_id"] % 10
    df = df.reset_index(drop=True)
    df["memmap_idx"] = df.index

    print(f"✅ Loaded {len(df)} records")
    return df

def prepare_master_cache(dataset_type, df_meta):
    cache_path = os.path.join(LOCAL_CACHE_DIR, f"{dataset_type}_master.dat")

    if os.path.exists(cache_path):
        print(f"✅ Cache found at {cache_path}")
        return cache_path

    print(f"🛠️ Building master cache: {dataset_type}")
    # Create the file on disk
    fp = np.memmap(
        cache_path,
        dtype="uint8",
        mode="w+",
        shape=(len(df_meta), 224, 224, 3)
    )

    path_to_idx = {row.path: i for i, row in df_meta.iterrows()}
    batch_files = [f for f in os.listdir(BATCH_DIR) if f.endswith(".pkl")]

    for bf in tqdm(batch_files, desc="Stitching batches"):
        with open(os.path.join(BATCH_DIR, bf), "rb") as f:
            batch = pickle.load(f).get(dataset_type, {})
            for path, img in batch.items():
                if path in path_to_idx:
                    idx = path_to_idx[path]
                    # Handle float 0-1 vs int 0-255
                    if img.max() <= 1.1:
                        img = (img * 255).astype("uint8")
                    
                    img = img.squeeze()
                    # Stack single channel 3 times for ResNet compatibility
                    fp[idx] = np.stack([img] * 3, axis=-1)

    fp.flush()
    del fp
    return cache_path

# --- 3. Dataset ---

class MimicStreamDataset(Dataset):
    def __init__(self, df, memmap, is_train=False):
        self.indices = df.memmap_idx.values
        self.labels = df.anchor_age.values.astype(np.float32)
        self.memmap = memmap
        self.is_train = is_train

        # Normalization for greyscale-as-RGB (same mean/std for all channels)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.485, 0.485],
            std=[0.229, 0.229, 0.229]
        )

        if is_train:
            self.augment = transforms.Compose([
                # Randomly shift and scale slightly to mimic lead placement variance
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05)
                ),
                # Note: Horizontal flip is intentionally OMITTED as ECGs 
                # have a specific time direction (P -> QRS -> T). 
                # Flipping time is physically invalid for age prediction.
            ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Zero-Copy read from Shared Memory
        img = self.memmap[self.indices[idx]]
        
        # Convert to Float Tensor (C, H, W)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.is_train:
            x = self.augment(x)

        x = self.normalize(x)
        y = torch.tensor(self.labels[idx]).view(1)
        return x, y

# --- 4. Training Engine ---

def evaluate(model, loader, device):
    """
    Returns the SUM of absolute errors and the count of samples.
    We do NOT divide by length here to safely aggregate across DDP.
    """
    model.eval()
    sum_mae = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda"):
                pred = model(x)
            
            # Clamp prediction to biological limits (0 to 120 years)
            pred = torch.clamp(pred, 0, 120)
            
            sum_mae += torch.abs(pred - y).sum().item()
    return sum_mae

def train_model(name, memmap_path, df, local_rank):
    is_main = local_rank == 0
    device = torch.device(f"cuda:{local_rank}")

    if is_main:
        print(f"\n{'='*40}\nTraining Model: {name}\n{'='*40}")

    # Splits
    train_df = df[df.fold < 8]
    val_df = df[df.fold == 8]
    test_df = df[df.fold == 9]

    # Load into Shared Memory (OS Page Cache)
    if is_main: print(f"⏳ Mapping {len(df)} images into Shared Memory...")
    shared_mem = np.memmap(
        memmap_path,
        dtype="uint8",
        mode="r",
        shape=(len(df), 224, 224, 3)
    )
    if is_main: print("✅ Memory map created.")

    # Datasets
    train_ds = MimicStreamDataset(train_df, shared_mem, is_train=True)
    val_ds = MimicStreamDataset(val_df, shared_mem, is_train=False)
    test_ds = MimicStreamDataset(test_df, shared_mem, is_train=False)

    # Samplers
    train_sampler = DistributedSampler(train_ds) if USE_DDP else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if USE_DDP else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if USE_DDP else None

    # Loaders
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=val_sampler,
                            shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, sampler=test_sampler,
                             shuffle=False, num_workers=NUM_WORKERS)

    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Adding Dropout for Regularization
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 1)
    )

    model.to(device)
    if USE_DDP:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer with Weight Decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda")

    # Plotting setup
    plt.switch_backend('agg')

    best_val = float("inf")
    history = {"train_mae": [], "val_mae": [], "epochs": []}

    for epoch in range(NUM_EPOCHS):
        if USE_DDP:
            train_sampler.set_epoch(epoch)

        model.train()
        running_mae = 0.0
        
        pbar = tqdm(train_loader, disable=not is_main, desc=f"Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                pred = model(x)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_mae += torch.abs(pred - y).sum().item()
            
            if is_main:
                pbar.set_postfix(mae=running_mae / ((pbar.n + 1) * BATCH_SIZE))

        # --- Metrics Aggregation (CORRECTED LOGIC) ---
        
        # 1. Get local sums
        val_sum_mae = evaluate(model, val_loader, device)
        train_sum_mae = running_mae
        
        # 2. Aggregate across GPUs
        if USE_DDP:
            t = torch.tensor([train_sum_mae, val_sum_mae], device=device)
            dist.all_reduce(t, dist.ReduceOp.SUM)
            train_sum_mae, val_sum_mae = t.tolist()

        # 3. Divide by Global Dataset Lengths to get true MAE
        avg_train_mae = train_sum_mae / len(train_df)
        avg_val_mae = val_sum_mae / len(val_df)

        scheduler.step(avg_val_mae)

        if is_main:
            print(f"Epoch {epoch+1}: Train MAE {avg_train_mae:.2f} | Val MAE {avg_val_mae:.2f}")
            history["train_mae"].append(avg_train_mae)
            history["val_mae"].append(avg_val_mae)
            history["epochs"].append(epoch + 1)

            # Save Plot
            plt.figure(figsize=(10, 6))
            plt.plot(history['epochs'], history['train_mae'], label='Train MAE')
            plt.plot(history['epochs'], history['val_mae'], label='Validation MAE')
            plt.title(f'Training Progress: {name}')
            plt.xlabel('Epochs')
            plt.ylabel('MAE (Years)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(RESULTS_DIR, f"{name}_loss_graph.png"))
            plt.close()

            # Save Checkpoint
            if avg_val_mae < best_val:
                best_val = avg_val_mae
                save_obj = model.module.state_dict() if USE_DDP else model.state_dict()
                torch.save(save_obj, os.path.join(CHECKPOINT_DIR, f"{name}_best.pth"))

    # Final Test Evaluation
    if is_main:
        print("\nLoading best model for testing...")
        model.load_state_dict(torch.load(
            os.path.join(CHECKPOINT_DIR, f"{name}_best.pth"),
            map_location=device
        ))
    
    # Evaluate Test Set (Using same aggregation logic)
    test_sum_mae = evaluate(model, test_loader, device)
    
    if USE_DDP:
        t = torch.tensor([test_sum_mae], device=device)
        dist.all_reduce(t, dist.ReduceOp.SUM)
        test_sum_mae = t.item()
    
    if is_main:
        final_test_mae = test_sum_mae / len(test_df)
        print(f"✅ FINAL TEST MAE: {final_test_mae:.2f} years")

        history["test_mae"] = final_test_mae
        with open(os.path.join(RESULTS_DIR, f"{name}_history.json"), "w") as f:
            json.dump(history, f, indent=4)

    # Cleanup
    del model, shared_mem, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()

# --- 5. Main ---

if __name__ == "__main__":
    local_rank = setup_ddp()

    if local_rank == 0:
        df_meta = load_mimic_metadata()
        master_path = prepare_master_cache(DATASET_TYPE, df_meta)
    else:
        df_meta = None
    
    if USE_DDP:
        dist.barrier()
        # Rank 0 has finished creating the cache, now safe for everyone to find the path
        if local_rank != 0:
            df_meta = load_mimic_metadata() # Load DF on all ranks so they know split sizes
            master_path = os.path.join(LOCAL_CACHE_DIR, f"{DATASET_TYPE}_master.dat")

    train_model(DATASET_TYPE, master_path, df_meta, local_rank)
    
    cleanup_ddp()
    if local_rank == 0:
        print("🎉 Training complete.")