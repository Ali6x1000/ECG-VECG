import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the cache file
cache_filepath = './vcg_only.pkl'

with open(cache_filepath, 'rb') as f:
    ecg_cache = pickle.load(f)

print(f"Cache contains {len(ecg_cache)} ECG images")
print("First few filenames:", list(ecg_cache.keys())[:5])

# Get the first image
first_filename = list(ecg_cache.keys())[9]
first_image = ecg_cache[first_filename]

print(f"Plotting image for: {first_filename}")
print(f"Image shape: {first_image.shape}")
print(f"Image data type: {first_image.dtype}")
print(f"Pixel value range: {first_image.min()} - {first_image.max()}")

# Plot the image
plt.figure(figsize=(10, 8))
# Remove the channel dimension for plotting (224, 224) instead of (1, 224, 224)
plt.imshow(first_image[0], cmap='gray', aspect='auto')
plt.title(f'ECG Image: {first_filename}')
plt.axis('off')  # Remove axes for cleaner view
plt.tight_layout()
plt.show()

# Alternative: Plot with axes to see pixel coordinates
plt.figure(figsize=(12, 8))
plt.imshow(first_image[0], cmap='gray', aspect='auto')
plt.title(f'ECG Image with Axes: {first_filename}')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.colorbar(label='Grayscale Value')
plt.tight_layout()
plt.savefig("test.png")