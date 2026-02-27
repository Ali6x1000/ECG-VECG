import os
import numpy as np
from tqdm import tqdm

class StatsCalculator:
    def __init__(self, training_mode, drive_base_path, img_size=(224, 224)):
        self.training_mode = training_mode
        self.img_size = img_size
        self.stats_path = os.path.join(drive_base_path, f"ptbxl_{training_mode}_stats.npz")
        
    def calculate_or_load_stats(self, local_image_cache_path, n_records):
        """Calculate normalization stats or load if already exists"""
        if os.path.exists(self.stats_path):
            stats = np.load(self.stats_path)
            channel_mean = stats['mean']
            channel_std = stats['std']
            print(f"✅ Per-channel stats for {self.training_mode} loaded.")
        else:
            print(f"🛠️ Global stats for {self.training_mode} not found. Calculating...")
            channel_mean, channel_std = self._calculate_stats(local_image_cache_path, n_records)
            
        print(f"Channel Mean: {channel_mean}")
        print(f"Channel Std: {channel_std}")
        return channel_mean, channel_std
    
    def _calculate_stats(self, local_image_cache_path, n_records):
        """Calculate channel-wise mean and standard deviation"""
        # Open the cache we just built
        image_cache = np.memmap(local_image_cache_path, dtype=np.float32, mode='r', 
                               shape=(n_records, *self.img_size, 3))

        pixel_sum = np.zeros(3, dtype=np.float64)
        pixel_sq_sum = np.zeros(3, dtype=np.float64)
        num_pixels = n_records * self.img_size[0] * self.img_size[1]

        # Iterate in chunks to save RAM
        for i in tqdm(range(0, n_records, 256), desc="Calculating Stats"):
            chunk = image_cache[i:i+256].astype(np.float64)
            pixel_sum += np.sum(chunk, axis=(0, 1, 2))
            pixel_sq_sum += np.sum(chunk**2, axis=(0, 1, 2))

        channel_mean = pixel_sum / num_pixels
        channel_std = np.sqrt(pixel_sq_sum / num_pixels - channel_mean**2)

        np.savez(self.stats_path, mean=channel_mean, std=channel_std)
        print(f"✅ Stats calculated and saved for {self.training_mode}.")
        del image_cache
        
        return channel_mean, channel_std
