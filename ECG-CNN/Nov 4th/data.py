import os
import pandas as pd
import numpy as np
import pickle
import keras
from tqdm import tqdm
from shutil import copyfile

class DataManager:
    def __init__(self, training_mode, drive_base_path, img_size=(224, 224), dtype=np.float32):
        self.training_mode = training_mode
        self.drive_base_path = drive_base_path
        self.img_size = img_size
        self.dtype = dtype
        
        # File paths
        self.meta_csv_path = os.path.join(drive_base_path, "ptbxl_database.csv")
        self.pickle_cache_path = ""
        self.local_pickle_cache_path = "/content/ecg_vcg_image_cache_hr_1x224x224.pkl"
        
        # Mode-specific cache paths
        self.image_cache_path = os.path.join(drive_base_path, f"ptbxl_{training_mode}_cache.npy")
        self.local_image_cache_path = f"/content/ptbxl_{training_mode}_cache.npy"
        
    def copy_data_from_drive(self):
        """Copy data from Drive to local Colab disk for speed"""
        if os.path.exists(self.drive_base_path):
            print("🚚 Copying data from Google Drive to local disk...")
            
            # Check if the mode-specific cache already exists on Drive
            if os.path.exists(self.image_cache_path):
                copyfile(self.image_cache_path, self.local_image_cache_path)
                print(f"✅ Found and copied existing {self.training_mode} cache.")
            else:
                print(f"⚠️ {self.training_mode} cache not found on Drive. It will be generated locally.")
        else:
            print("⚠️ Google Drive not mounted or path is incorrect. Using local paths.")
            self.local_pickle_cache_path = self.pickle_cache_path
    
    def load_metadata(self):
        """Load full metadata from CSV"""
        print("Loading metadata...")
        try:
            y_full = pd.read_csv(self.meta_csv_path, index_col="ecg_id")
        except FileNotFoundError:
            print(f"❌ ERROR: Metadata file not found at {self.meta_csv_path}")
            print("Please make sure 'ptbxl_database.csv' is in your Drive folder.")
            raise
            
        n_records = len(y_full)
        print(f"✅ Loaded metadata for {n_records} records.")
        return y_full, n_records
    
    def generate_image_cache(self, y_full, n_records):
        """Generate mode-specific image cache from pickle file"""
        if os.path.exists(self.local_image_cache_path):
            print(f"✅ {self.training_mode} image cache found. Skipping image generation.")
            return
            
        print(f"🛠️ {self.training_mode} image cache not found. Generating from pickle file...")
        
        # Load the source pickle data
        try:
            with open(self.local_pickle_cache_path, 'rb') as f:
                pickle_data = pickle.load(f)
        except FileNotFoundError:
            print(f"❌ ERROR: Local pickle file not found at {self.local_pickle_cache_path}")
            raise
            
        print(f"✅ Loaded pickle file with {len(pickle_data)} entries.")

        # Create a writeable memmap for the new cache
        image_cache = np.memmap(self.local_image_cache_path, dtype=self.dtype, mode='w+', 
                               shape=(n_records, *self.img_size, 3))
        
        valid_filenames = set(pickle_data.keys())
        missing_count = 0

        for i, (ecg_id, row) in enumerate(tqdm(y_full.iterrows(), total=n_records, 
                                              desc=f"Building {self.training_mode} cache")):
            filename = row["filename_hr"]
            
            # Check if we have data for this row
            if pd.isna(filename) or filename not in valid_filenames:
                missing_count += 1
                # Store a blank image if no data
                image_cache[i] = np.zeros((*self.img_size, 3), dtype=self.dtype)
                continue

            # Load images, squeeze (1, 224, 224) -> (224, 224)
            ecg_img = pickle_data[filename]['ecg'].squeeze().astype(self.dtype)
            vcg_img = pickle_data[filename]['vcg'].squeeze().astype(self.dtype)
            
            final_image = np.zeros((*self.img_size, 3), dtype=self.dtype)

            # Assemble the 3-channel image based on the mode
            if self.training_mode == 'ecg_only':
                final_image[:, :, 0] = ecg_img
                final_image[:, :, 1] = ecg_img
                final_image[:, :, 2] = ecg_img
            elif self.training_mode == 'vcg_only':
                final_image[:, :, 0] = vcg_img
                final_image[:, :, 1] = vcg_img
                final_image[:, :, 2] = vcg_img
            elif self.training_mode == 'combined':
                avg_img = (ecg_img + vcg_img) / 2.0
                final_image[:, :, 0] = ecg_img
                final_image[:, :, 1] = vcg_img
                final_image[:, :, 2] = avg_img

            image_cache[i] = final_image

        image_cache.flush()
        del image_cache
        print(f"✅ {self.training_mode} cache generated.")
        if missing_count > 0:
            print(f"⚠️ Warning: {missing_count} records from metadata had no matching entry in the pickle file.")

class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, image_cache_path, labels, indices, batch_size, mean, std, img_size, n_records):
        self.indices = indices
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.labels = labels
        self.image_cache_path = image_cache_path
        
        # Preload data for this split into RAM
        print(f"🧠 Preloading {len(indices)} samples for this generator...")
        # Open the full cache
        full_image_data_memmap = np.memmap(self.image_cache_path, dtype=np.float32, mode='r', 
                                          shape=(n_records, *img_size, 3))
        # Select only the indices for this split (train/val/test) and load into RAM
        self.X_cached = full_image_data_memmap[self.indices]
        self.y_cached = self.labels[self.indices]
        del full_image_data_memmap
        print("✅ Preloading complete.")

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.indices))

        # Get preloaded batch
        X_batch = self.X_cached[start:end]
        y_batch = self.y_cached[start:end]

        # Apply normalization
        X_batch = (X_batch - self.mean) / (self.std + 1e-7)
        return X_batch, y_batch
