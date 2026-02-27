#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import shutil
from collections import defaultdict


# Configuration
CSV_PATH = "/condo/alkindilab/shared/Mohammad/ECG/Age_Prediction/Data_without_CAC_BNP/New_Data/all_ecg_cropped.csv"
BATCH_SIZE = 8
EPOCHS = 1000
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
EARLY_STOPPING_PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "/condo/alkindilab/shared/Mohammad/ECG/Age_Prediction/9_6/resnet50_results/"
SAVE_AUGMENTED_SAMPLES = False
AUGMENTED_SAMPLES_DIR = os.path.join(RESULTS_DIR, "augmentation_samples")
NUM_SAMPLES_TO_SAVE = 20
AUGMENTATION_SAMPLES_PER_IMAGE = 9  # Number of augmented versions to save per original

os.makedirs(RESULTS_DIR, exist_ok=True)
if SAVE_AUGMENTED_SAMPLES:
    os.makedirs(AUGMENTED_SAMPLES_DIR, exist_ok=True)

# ========== RANDOM ZOOM CLASS ==========

class RandomZoom:
    """Custom random zoom augmentation for medical images"""
    def __init__(self, scale_range: tuple = (0.7, 1.2), target_size: tuple = (224, 224)):
        self.scale_range = scale_range
        self.target_size = target_size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        new_w, new_h = int(w * scale), int(h * scale)
        img = transforms.functional.resize(img, (new_h, new_w))
        padding = self._get_padding(img)
        if any(padding):
            img = transforms.functional.pad(img, padding, fill=255)
        img = transforms.functional.center_crop(img, self.target_size)
        return img

    def _get_padding(self, img: Image.Image) -> tuple:
        w, h = img.size
        pad_w = max(0, (self.target_size[0] - w + 1) // 2)
        pad_h = max(0, (self.target_size[1] - h + 1) // 2)
        return (pad_w, pad_h, pad_w, pad_h)

# ========== MRN STRATIFIED SPLIT FUNCTION ==========

def get_mrn_stratified_split(df, test_size=0.2, val_size=0.2, random_state=42):
    """Split data into train, validation, and test sets with no MRN overlap"""
    # Reset index to ensure we have clean 0-based indexing
    df = df.reset_index(drop=True)
    
    # Group by MRN and collect all records for each MRN
    mrn_groups = defaultdict(list)
    for idx, row in df.iterrows():
        mrn_groups[row['MRN']].append(idx)

    # Create a representative dataframe with one row per MRN
    mrn_df = pd.DataFrame({
        'MRN': list(mrn_groups.keys()),
        'Age': [df.loc[ids[0], 'Age'] for ids in mrn_groups.values()],
        'indices': [ids for ids in mrn_groups.values()]
    })

    # Create age bins for stratification
    age_bins = [18, 40, 60, 80, 100]  # Added upper bound
    mrn_df['age_bin'] = pd.cut(mrn_df['Age'], bins=age_bins, labels=False, include_lowest=True)
    
    # Handle any NaN values in age_bin
    mrn_df = mrn_df.dropna(subset=['age_bin'])
    
    # First split: separate test set
    train_val_mrn_df, test_mrn_df = train_test_split(
        mrn_df, test_size=test_size, stratify=mrn_df['age_bin'], random_state=random_state
    )
    
    # Second split: separate validation set from train+val
    val_proportion = val_size / (1 - test_size)
    train_mrn_df, val_mrn_df = train_test_split(
        train_val_mrn_df, test_size=val_proportion, 
        stratify=train_val_mrn_df['age_bin'], random_state=random_state
    )
    
    # Convert MRN indices to original indices - FIXED
    train_records = []
    for indices_list in train_mrn_df['indices']:
        train_records.extend(indices_list)
    
    val_records = []
    for indices_list in val_mrn_df['indices']:
        val_records.extend(indices_list)
    
    test_records = []
    for indices_list in test_mrn_df['indices']:
        test_records.extend(indices_list)
    
    return train_records, val_records, test_records

# ========== NAMED TRANSFORM CLASS ==========

class NamedTransform:
    """Wrapper for transforms to give them identifiable names"""
    def __init__(self, name, transform):
        self.name = name
        self.transform = transform
        
    def __call__(self, img):
        return self.transform(img)
    
    def __repr__(self):
        return f"NamedTransform({self.name}, {self.transform})"

# ========== UTILITY FUNCTIONS ==========
def print_model_info(model, input_size=(3, 224, 224), batch_size=BATCH_SIZE):
    """Print detailed information about the model"""
    print(f"\n{'='*50}")
    print("MODEL ARCHITECTURE DETAILS")
    print(f"{'='*50}")
    
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Model size estimation
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"Estimated model size: {size_all_mb:.2f} MB")
    
    # Memory usage estimation for inference
    try:
        # Create a dummy input
        dummy_input = torch.randn(batch_size, *input_size, device=DEVICE)
        
        # Move model to device first
        model_device = next(model.parameters()).device
        if model_device != DEVICE:
            model = model.to(DEVICE)
            
        # Forward pass to get memory usage
        with torch.no_grad():
            model(dummy_input)
        
        if torch.cuda.is_available():
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated(DEVICE) / 1024**2:.2f} MB")
            print(f"GPU Memory cached: {torch.cuda.memory_reserved(DEVICE) / 1024**2:.2f} MB")
    
    except Exception as e:
        print(f"Memory estimation failed: {e}")
    
    # Layer-wise information
    print(f"\nLayer-wise breakdown:")
    print(f"{'Layer Type':<25} {'Parameters':<15} {'Output Shape':<20}")
    print(f"{'-'*25} {'-'*15} {'-'*20}")
    
    # Sample some key layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            params = sum(p.numel() for p in module.parameters())
            if hasattr(module, 'out_channels'):
                output_shape = f"(..., {module.out_channels})"
            elif hasattr(module, 'out_features'):
                output_shape = f"(..., {module.out_features})"
            else:
                output_shape = "N/A"
            
            print(f"{name[:24]:<25} {params:<15,} {output_shape:<20}")
    
    # Model architecture summary
    print(f"\nModel classifier layers:")
    if hasattr(model, 'fc'):
        # Handle both Linear and Sequential fc layers
        if isinstance(model.fc, nn.Sequential):
            print(f"  Final Layers (Sequential):")
            for i, layer in enumerate(model.fc):
                if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                    print(f"    Layer {i}: {layer.__class__.__name__} - {layer.in_features} -> {layer.out_features}")
                elif hasattr(layer, 'weight'):
                    print(f"    Layer {i}: {layer.__class__.__name__}")
        else:
            # Single Linear layer
            if hasattr(model.fc, 'in_features') and hasattr(model.fc, 'out_features'):
                print(f"  Final Layer: {model.fc.__class__.__name__} - {model.fc.in_features} -> {model.fc.out_features}")
    
    print(f"{'='*50}\n")
    
def save_data_splits(df, split_name):
    df.to_csv(os.path.join(RESULTS_DIR, f"{split_name}_split.csv"), index=False)

def save_test_results(test_df, predictions, true_values):
    results_df = test_df.copy()
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    results_df['predicted_age'] = predictions
    results_df['true_age'] = true_values
    results_df['absolute_error'] = np.abs(predictions - true_values)
    results_df['squared_error'] = (predictions - true_values)**2
    
    results_df.to_csv(os.path.join(RESULTS_DIR, "test_results_details.csv"), index=False)
        
    pearson_r, pearson_p = pearsonr(true_values, predictions)
    
    summary_stats = {
        'RMSE': [np.sqrt(mean_squared_error(true_values, predictions))],
        'R2': [r2_score(true_values, predictions)],
        'MAE': [np.mean(np.abs(predictions - true_values))],
        'Max_Error': [np.max(np.abs(predictions - true_values))],
        'Pearson_R': [pearson_r],
        'Pearson_p_value': [pearson_p]
    }
    pd.DataFrame(summary_stats).to_csv(os.path.join(RESULTS_DIR, "test_results_summary.csv"), index=False)
    
    print("Saved test results to CSV files")

def plot_predicted_vs_real(true_ages, predicted_ages):
    plt.figure(figsize=(10, 8))
    plt.scatter(true_ages, predicted_ages, alpha=0.6, edgecolors='w', linewidth=0.5)
    
    m, b = np.polyfit(true_ages, predicted_ages, 1)
    plt.plot(true_ages, m*np.array(true_ages)+b, 'r-', linewidth=2, 
             label=f'Regression: y = {m:.2f}x + {b:.2f}')
    
    min_age = min(min(true_ages), min(predicted_ages))
    max_age = max(max(true_ages), max(predicted_ages))
    plt.plot([min_age, max_age], [min_age, max_age], 'k--', label='Perfect prediction')
    
    r, p = pearsonr(true_ages, predicted_ages)
    rmse = np.sqrt(mean_squared_error(true_ages, predicted_ages))
    mae = np.mean(np.abs(np.array(predicted_ages) - np.array(true_ages)))
    
    p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    stats_text = (f'Pearson R = {r:.3f} ({p_text})\n'
                  f'RMSE = {rmse:.2f} years\n'
                  f'MAE = {mae:.2f} years')
    
    plt.annotate(stats_text, xy=(0.05, 0.80), xycoords='axes fraction',
                 fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('True Age (years)', fontsize=12)
    plt.ylabel('Predicted Age (years)', fontsize=12)
    plt.title('ECG-Based Age Prediction: Predicted vs True Age', fontsize=14, pad=20)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(RESULTS_DIR, "age_prediction_scatter")
    
    plt.savefig(f"{plot_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{plot_path}.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved predicted vs real age plot to {plot_path}.[png/pdf]")

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_rmse'], label='Validation RMSE', color='orange')
    plt.plot(history['test_rmse'], label='Test RMSE', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Validation and Test RMSE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    plot_path = os.path.join(RESULTS_DIR, "training_history.png")
    
    plt.savefig(plot_path)
    plt.close()
    print("Saved training history plot")

# ========== DATASET CLASS ==========

class ECGDataset(Dataset):
    def __init__(self, dataframe, transform=None, train_mode=False):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.train_mode = train_mode
        self.samples_saved = False
        
        if self.train_mode and SAVE_AUGMENTED_SAMPLES:
            os.makedirs(AUGMENTED_SAMPLES_DIR, exist_ok=True)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["cropped_image_path"]
        age = self.data.iloc[idx]["Age"]
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Verify image size meets minimum requirements
            if image.size[0] < 50 or image.size[1] < 50:
                print(f"Warning: Very small image detected at {img_path} with size {image.size}")
                # Create a blank image as fallback
                image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='white')
            
            if self.train_mode and SAVE_AUGMENTED_SAMPLES and not self.samples_saved and idx < NUM_SAMPLES_TO_SAVE:
                sample_dir = os.path.join(AUGMENTED_SAMPLES_DIR, f'sample_{idx}')
                os.makedirs(sample_dir, exist_ok=True)
                
                # Save original
                original_path = os.path.join(sample_dir, 'original.png')
                image.save(original_path)
                
                # Save individual augmentation steps
                self._save_augmentation_steps(image, sample_dir)
                
                # Save multiple random augmented versions
                for i in range(AUGMENTATION_SAMPLES_PER_IMAGE):
                    aug_img = image.copy()
                    for t in self.transform.transforms[:-1]:  # Exclude normalization
                        aug_img = t(aug_img)
                    
                    if isinstance(aug_img, torch.Tensor):
                        aug_img = transforms.ToPILImage()(aug_img)
                    augmented_path = os.path.join(sample_dir, f'augmented_{i}.png')
                    aug_img.save(augmented_path)
                
                if idx == NUM_SAMPLES_TO_SAVE - 1:
                    self.samples_saved = True
            
            if self.transform:
                image = self.transform(image)
                
            return image, torch.tensor(age, dtype=torch.float)
        
        except Exception as e:
            print(f"Error loading image: {img_path}")
            print(f"Error: {str(e)}")
            # Return a blank image and mean age as fallback
            mean_age = self.data['Age'].mean()
            blank_image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='white')
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, torch.tensor(mean_age, dtype=torch.float)
    
    def _save_augmentation_steps(self, image, sample_dir):
        """Save each augmentation step separately for visualization"""
        current_img = image.copy()
        
        # We'll track the sequence of applied transforms
        transform_sequence = []
        
        for i, t in enumerate(self.transform.transforms[:-1]):  # Exclude normalization
            # Apply the current transform
            new_img = t(current_img.copy())  # Apply to copy to isolate effect
            
            # Get transform name
            transform_name = getattr(t, 'name', t.__class__.__name__)
            if isinstance(t, transforms.RandomApply):
                transform_name = f"RandomApply_{getattr(t.transforms[0], 'name', t.transforms[0].__class__.__name__)}"
            
            # Save before and after if it's a transform we want to track
            if isinstance(t, (transforms.ColorJitter,
                            transforms.RandomGrayscale,
                            transforms.RandomResizedCrop,
                            transforms.CenterCrop,
                            transforms.GaussianBlur,
                            transforms.RandomApply,
                            RandomZoom)):  # Added RandomZoom to tracking
            
                # Save the image before this transform
                before_path = os.path.join(sample_dir, f'before_{transform_name}.png')
                if isinstance(current_img, torch.Tensor):
                    transforms.ToPILImage()(current_img).save(before_path)
                else:
                    current_img.save(before_path)
                
                # Save the image after this transform
                after_path = os.path.join(sample_dir, f'after_{transform_name}.png')
                if isinstance(new_img, torch.Tensor):
                    transforms.ToPILImage()(new_img).save(after_path)
                else:
                    new_img.save(after_path)
                
                transform_sequence.append(transform_name)
            
            current_img = new_img
        
        # Save the transform sequence as a text file
        with open(os.path.join(sample_dir, 'transform_sequence.txt'), 'w') as f:
            f.write("Transform sequence applied:\n")
            f.write("\n".join(transform_sequence))

# ========== TRAINING FUNCTION ==========

def train_and_evaluate(train_df, val_df, test_df):
    # Save data splits
    save_data_splits(train_df, "train")
    save_data_splits(val_df, "val")
    save_data_splits(test_df, "test")
    
    # Base transform (applied to all images)
    base_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # First operation to ensure consistent size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Training transform with augmentations including RandomZoom
    train_transform = transforms.Compose([
        # First ensure consistent size
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        
        # Color transforms
        NamedTransform(
            name="ColorJitter",
            transform=transforms.ColorJitter(
                brightness=0.08,
                contrast=0.08,
                saturation=0.05,
                hue=0.005
            )
        ),
        NamedTransform(
            name="RandomGrayscale",
            transform=transforms.RandomGrayscale(p=0.1)
        ),
        
        # Blur
        NamedTransform(
            name="RandomBlur",
            transform=transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.2)
        ),
        
        # Geometric transforms - REPLACED with RandomZoom
        NamedTransform(
            name="RandomZoom",
            transform=transforms.RandomApply([
                RandomZoom(scale_range=(0.8, 1.2), target_size=(IMAGE_SIZE, IMAGE_SIZE))
            ], p=0.6)  # Higher probability since it's the main geometric augmentation
        ),
        
        # Final base transforms
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets with error handling
    train_dataset = ECGDataset(train_df, transform=train_transform, train_mode=True)
    val_dataset = ECGDataset(val_df, transform=base_transform)
    test_dataset = ECGDataset(test_df, transform=base_transform)

    # Reduced number of workers to avoid system warnings
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    print(f"\n{'='*60}\nStarting Training\n{'='*60}")
    
    # Load ResNet50 with pretrained weights
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Modify the final fully connected layer for regression (1 output)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1)
    )
    
    # Move model to device before printing info
    model = model.to(DEVICE)
    print_model_info(model, input_size=(3, IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE)

    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_rmse = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_rmse': [], 'test_rmse': [], 'lr': []}

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training")
        
        for images, labels in train_loader_tqdm:
            # Skip problematic batches
            if images.size(0) == 0:
                continue
                
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_loader_tqdm.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Evaluate on validation set for early stopping
        model.eval()
        val_preds, val_labels = [], []
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation")
        with torch.no_grad():
            for images, labels in val_loader_tqdm:
                if images.size(0) == 0:
                    continue
                    
                images = images.to(DEVICE, non_blocking=True)
                outputs = model(images).cpu().numpy().flatten()
                val_preds.extend(outputs)
                val_labels.extend(labels.numpy())

        val_rmse = np.sqrt(mean_squared_error(val_labels, val_preds))
        history['val_rmse'].append(val_rmse)
        
        # Also evaluate on test set for monitoring (but don't use for early stopping)
        test_preds, test_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                if images.size(0) == 0:
                    continue
                    
                images = images.to(DEVICE, non_blocking=True)
                outputs = model(images).cpu().numpy().flatten()
                test_preds.extend(outputs)
                test_labels.extend(labels.numpy())

        test_rmse = np.sqrt(mean_squared_error(test_labels, test_preds))
        history['test_rmse'].append(test_rmse)
        
        scheduler.step(val_rmse)
        
        val_r2 = r2_score(val_labels, val_preds)
        val_pearson_r = pearsonr(val_labels, val_preds)[0]
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.2f} | "
              f"Test RMSE: {test_rmse:.2f} | Val R2: {val_r2:.3f} | Val Pearson R: {val_pearson_r:.3f} | LR: {history['lr'][-1]:.2e}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_rmse': best_val_rmse,
            }, os.path.join(RESULTS_DIR, "best_model.pth"))
            print("Model improved and saved")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    pd.DataFrame(history).to_csv(os.path.join(RESULTS_DIR, "training_history.csv"), index=False)
    plot_training_history(history)

    print(f"\nEvaluating best model on test set...")
    checkpoint = torch.load(os.path.join(RESULTS_DIR, "best_model.pth"), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_preds, test_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            if images.size(0) == 0:
                continue
                
            images = images.to(DEVICE)
            outputs = model(images).cpu().numpy().flatten()
            test_preds.extend(outputs)
            test_labels.extend(labels.numpy())

    test_rmse = np.sqrt(mean_squared_error(test_labels, test_preds))
    test_r2 = r2_score(test_labels, test_preds)
    test_pearson_r, test_pearson_p = pearsonr(test_labels, test_preds)
    
    print(f"\nFinal Test Results:")
    print("RMSE: {:.2f}".format(test_rmse))
    print("R2 Score: {:.4f}".format(test_r2))
    print("Pearson R: {:.4f} (p-value: {:.4e})".format(test_pearson_r, test_pearson_p))
    print("MAE: {:.2f}".format(np.mean(np.abs(np.array(test_preds) - np.array(test_labels)))))
    
    save_test_results(test_df, test_preds, test_labels)
    plot_predicted_vs_real(test_labels, test_preds)
    
    return {
        'rmse': test_rmse,
        'r2': test_r2,
        'pearson_r': test_pearson_r,
        'mae': np.mean(np.abs(np.array(test_preds) - np.array(test_labels)))
    }

# ========== MAIN SCRIPT ==========

if __name__ == "__main__":
    print("="*60)
    print("ECG Age Prediction Model Training")
    print("="*60)
    print("\nLoading and splitting data with MRN-stratified split...")

    # Load initial DataFrame
    df_full = pd.read_csv(CSV_PATH)

    # Initial Cleaning (apply to the full dataset before splitting)
    df_full = df_full.dropna(subset=["Age", "cropped_image_path"])
    df_full = df_full[df_full["cropped_image_path"].apply(os.path.exists)]
    df_full = df_full[(df_full["Age"] >= 18) & (df_full["Age"] <= 80)]  # Age filtering
    
    print(f"Total records after initial cleaning: {len(df_full)}")

    try:
        train_indices, val_indices, test_indices = get_mrn_stratified_split(
            df_full, test_size=0.05, val_size=0.01, random_state=42
        )
        
        # Verify indices are within bounds
        max_index = len(df_full) - 1
        for idx_list, name in [(train_indices, 'train'), (val_indices, 'val'), (test_indices, 'test')]:
            if idx_list:
                if max(idx_list) > max_index or min(idx_list) < 0:
                    print(f"ERROR: {name} indices out of bounds!")
                    print(f"Max allowed: {max_index}, Found: min={min(idx_list)}, max={max(idx_list)}")
                    # Fall back to simple split
                    raise ValueError("Indices out of bounds")
        
        train_df = df_full.iloc[train_indices].reset_index(drop=True)
        val_df = df_full.iloc[val_indices].reset_index(drop=True)
        test_df = df_full.iloc[test_indices].reset_index(drop=True)
        
    except Exception as e:
        print(f"Error in stratified split: {e}")
        print("Falling back to simple MRN-based split...")
        
        # Fallback to original simple split method
        mrn_counts = df_full['MRN'].value_counts()
        truly_unique_mrns = set(mrn_counts[mrn_counts == 1].index)
        
        test_df = df_full[df_full['MRN'].isin(truly_unique_mrns)]
        train_val_df = df_full[~df_full['MRN'].isin(truly_unique_mrns)]
        
        # Split train_val into train and validation
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.2, stratify=train_val_df['Age'], random_state=42
        )
        
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    # Sanity check: Ensure no MRN overlap
    train_mrns = set(train_df['MRN'].unique())
    val_mrns = set(val_df['MRN'].unique())
    test_mrns = set(test_df['MRN'].unique())
    
    overlap_train_val = train_mrns.intersection(val_mrns)
    overlap_train_test = train_mrns.intersection(test_mrns)
    overlap_val_test = val_mrns.intersection(test_mrns)
    
    if any([overlap_train_val, overlap_train_test, overlap_val_test]):
        print(f"WARNING: MRN overlap detected!")
        if overlap_train_val: print(f"  Train-Val overlap: {overlap_train_val}")
        if overlap_train_test: print(f"  Train-Test overlap: {overlap_train_test}")
        if overlap_val_test: print(f"  Val-Test overlap: {overlap_val_test}")
    else:
        print("Confirmed: No MRN overlap between any splits.")

    # Display data split information
    print(f"\n{'='*40}")
    print("Data Split Information (MRN-stratified split)")
    print(f"{'='*40}")
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    total_samples = len(train_df) + len(val_df) + len(test_df)
    if total_samples > 0:
        print(f"Training set: {100 * len(train_df) / total_samples:.1f}%")
        print(f"Validation set: {100 * len(val_df) / total_samples:.1f}%")
        print(f"Test set: {100 * len(test_df) / total_samples:.1f}%")
    
    print("\nTraining set age statistics:")
    print(train_df['Age'].describe())
    print("\nValidation set age statistics:")
    print(val_df['Age'].describe())
    print("\nTest set age statistics:")
    print(test_df['Age'].describe())
    
    # Display device information
    print(f"\nUsing device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Display model configuration
    print("\nModel Configuration:")
    print(f"- Architecture: ResNet50")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Learning rate: {LEARNING_RATE}")
    print(f"- Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"- Epochs: {EPOCHS}")
    print(f"- Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs")
    
    # Print augmentation information
    print("\nAugmentation Configuration:")
    print("- Color Jitter (brightness=0.08, contrast=0.08, saturation=0.05, hue=0.005)")
    print("- Random Grayscale (p=0.1)")
    print("- Random Blur (p=0.2)")
    print("- Random Zoom (p=0.6, scale_range=(0.8, 1.2))")
    print(f"- Saving augmentation samples: {'Yes' if SAVE_AUGMENTED_SAMPLES else 'No'}")
    if SAVE_AUGMENTED_SAMPLES:
        print(f"- Number of samples to save: {NUM_SAMPLES_TO_SAVE}")
        print(f"- Augmented versions per sample: {AUGMENTATION_SAMPLES_PER_IMAGE}")
    
    # Train and evaluate
    result = train_and_evaluate(train_df, val_df, test_df)
    
    print("\nTraining and evaluation complete!")
    print(f"All results saved to: {RESULTS_DIR}")