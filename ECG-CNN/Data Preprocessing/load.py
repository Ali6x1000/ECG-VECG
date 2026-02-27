import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import wfdb
import matplotlib
import matplotlib.pyplot as plt
import cv2
import pickle
from scipy.signal import butter, filtfilt
import sys
import concurrent.futures
import multiprocessing
# Update this path to your specific library location
sys.path.append('/Users/alinawaf/Desktop/Research/ECG-VECG/CardioVectorLib')
from cardiovector import plotting, preprocessing, reconstruction as rec

# --- Matplotlib Configuration ---
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9.5

# --- Constants ---
MM_PER_MV = 10
MM_PER_SEC = 25
SAMPLING_RATE = 500

# --- 1. Signal Processing Helpers (Unchanged logic, just organized) ---

def apply_baseline_correction(df: pd.DataFrame, sampling_rate: float = SAMPLING_RATE, cutoff_hz: float = 0.67) -> pd.DataFrame:
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(N=2, Wn=normal_cutoff, btype='high', analog=False)
    filtered_df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if not df[col].isnull().all() and df[col].std() > 1e-6:
                filtered_df[col] = filtfilt(b, a, df[col].values)
    return filtered_df

def get_vcg_signals_from_record(record: wfdb.Record) -> np.ndarray:
    """
    Extracts 3-lead VCG signals using CardioVectorLib logic.
    Returns np.ndarray (N, 3) or None.
    """
    try:
        record_copy = record
        record_copy.sig_name = [name.lower() for name in record.sig_name]
        record_copy.base_datetime = None
        
        # 1. Baseline Correction
        try:
            filtered_record = preprocessing.remove_baseline_wandering(record_copy)
        except:
            # Fallback manual filtering
            df = pd.DataFrame(record.p_signal, columns=record.sig_name)
            nyquist = 0.5 * SAMPLING_RATE
            normal_cutoff = 0.67 / nyquist
            b, a = butter(N=2, Wn=normal_cutoff, btype='high', analog=False)
            filtered_data = df.copy()
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]) and not df[col].isnull().all():
                    if df[col].std() > 1e-6:
                        filtered_data[col] = filtfilt(b, a, df[col].values)
            filtered_record = record_copy
            filtered_record.p_signal = filtered_data.values
            filtered_record.sig_name = [name.lower() for name in record.sig_name]

        # 2. Kors Transformation
        vcg_record = rec.kors_vcg(filtered_record)
        
        if vcg_record is None: return None
            
        if not hasattr(vcg_record, 'p_signal') or vcg_record.p_signal is None:
            if hasattr(vcg_record, 'd_signal') and vcg_record.d_signal is not None:
                vcg_signals = vcg_record.d_signal.astype(float)
            else:
                return None
        else:
            vcg_signals = vcg_record.p_signal
        
        if vcg_signals.shape[1] != 3: return None
        return vcg_signals

    except Exception as e:
        # print(f"VCG calc error: {e}")
        return None

# --- 2. Plotting Helpers ---

def _plot_vcg_loop_simple(ax: plt.Axes, vcg_signals: np.ndarray, projection='frontal'):
    """Plots a single clean VCG loop on the provided axis."""
    vx, vy, vz = vcg_signals.T
    
    if projection == 'frontal': proj_x, proj_y = vx, vy
    elif projection == 'horizontal': proj_x, proj_y = vx, vz
    elif projection == 'sagittal': proj_x, proj_y = vy, vz
    else: proj_x, proj_y = vx, vy 
    
    ax.plot(proj_x, proj_y, 'black', linewidth=1.2, alpha=0.9)
    ax.axis('equal')
    ax.grid(False)
    ax.axis('off')

def _plot_ecg_leads_on_axis(ax: plt.Axes, ecg_data: pd.DataFrame):
    """Plots standard 12-lead ECG on a single large axis."""
    leads_layout = [
        ['I',  'AVR', 'V1', 'V4'],
        ['II', 'AVL', 'V2', 'V5'],
        ['III','AVF', 'V3', 'V6']
    ]
    
    data = ecg_data.copy()
    total_samples = len(data)
    total_columns = 4
    segment_length = total_samples // total_columns
    
    # Setup Axis Limits
    # Assuming typical height for 3 rows of signals
    y_max = 12 # Arbitrary scaling unit to fit rows
    ax.set_ylim(0, y_max)
    ax.set_xlim(0, total_samples)
    ax.axis('off') # Hide box

    total_rows = 3
    y_offset_step = y_max / total_rows

    for row_idx, lead_row in enumerate(leads_layout):
        for col_idx, lead in enumerate(lead_row):
            if lead not in data.columns or data[lead].isnull().all(): continue
            
            seg_start = col_idx * segment_length
            seg_end = (col_idx + 1) * segment_length
            
            # Calculate vertical offset
            offset = (total_rows - row_idx - 0.5) * y_offset_step
            
            # Plot
            segment = data[lead].values[seg_start:seg_end]
            # Scaling factor 0.5 to fit signals nicely in rows
            ax.plot(np.arange(seg_start, seg_end), (segment * 0.5) + offset, 
                    linewidth=0.6, color='black')

# --- 3. Master Combined Image Generator ---

def generate_combined_image(record: wfdb.Record, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Generates a single image with:
    - Top 60%: 12-Lead ECG
    - Bottom 40%: 3-View VCG (Frontal, Horizontal, Sagittal)
    """
    # 1. Prepare Data
    df = pd.DataFrame(record.p_signal, columns=record.sig_name)
    ecg_data = apply_baseline_correction(df)
    
    vcg_signals = get_vcg_signals_from_record(record)
    
    if ecg_data.empty or vcg_signals is None:
        return None

    # 2. Create Figure with GridSpec (Height Ratios 6:4)
    # Using a slightly taller figsize to accommodate the stack
    fig = plt.figure(figsize=(10, 10), dpi=100)
    fig.set_facecolor('white')
    
    # GridSpec: 2 Rows (60% vs 40%), 3 Columns (for VCG split)
    gs = fig.add_gridspec(2, 3, height_ratios=[6, 4], 
                          hspace=0.05, wspace=0.05,
                          left=0.02, right=0.98, top=0.98, bottom=0.02)

    # --- TOP PART: ECG (Spans all 3 columns) ---
    ax_ecg = fig.add_subplot(gs[0, :])
    ax_ecg.set_facecolor('white')
    _plot_ecg_leads_on_axis(ax_ecg, ecg_data)

    # --- BOTTOM PART: VCG (Split into 3 columns) ---
    ax_vcg_f = fig.add_subplot(gs[1, 0]) # Frontal
    ax_vcg_h = fig.add_subplot(gs[1, 1]) # Horizontal
    ax_vcg_s = fig.add_subplot(gs[1, 2]) # Sagittal

    _plot_vcg_loop_simple(ax_vcg_f, vcg_signals, projection='frontal')
    _plot_vcg_loop_simple(ax_vcg_h, vcg_signals, projection='horizontal')
    _plot_vcg_loop_simple(ax_vcg_s, vcg_signals, projection='sagittal')

    # 3. Render and Post-Process
    fig.canvas.draw()
    img_buf = fig.canvas.buffer_rgba()
    img_array_rgba = np.asarray(img_buf)
    plt.close(fig)

    # Convert to Grayscale
    img_array_bgr = cv2.cvtColor(img_array_rgba, cv2.COLOR_RGBA2BGR)
    img_array_gray = cv2.cvtColor(img_array_bgr, cv2.COLOR_BGR2GRAY)
    
    # Resize to Target
    img_resized = cv2.resize(img_array_gray, target_size, interpolation=cv2.INTER_AREA)
    img_resized[img_resized < 240] = 0
    # Reshape to (1, H, W)
    img_final = np.expand_dims(img_resized, axis=0)
    
    return img_final

# --- 4. Main Execution Loop ---
# --- 4. Main Execution Loop ---

# # Local Path Configuration
# folderpath = '/Users/alinawaf/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'

# df_full = pd.read_csv(folderpath + "ptbxl_database.csv")
# df_full = df_full.dropna(subset=["age", "filename_hr"])
# df_full = df_full[(df_full["age"] >= 18) & (df_full["age"] <= 80)]

# ecg_cache = {}
# vcg_cache = {}
# combined_cache = {}
# invalid_paths = []

# print("Starting generation of ECG, VCG, and Combined Images...")

# # Processing Loop
# # Adjusted to run on head(5) for testing, remove .head(5) for full run
# for _, row in tqdm(df_full.head(5).iterrows(), total=5):
#     filename = row["filename_hr"] 
#     full_path = os.path.join(folderpath, filename)
    
#     try:
#         record = wfdb.rdrecord(full_path)
        
#         # Generate the combined image
#         combined_image = generate_combined_image(record, target_size=(224, 224))
        
#         if combined_image is not None:
#             # Row 130 separates ECG (top) from VCG (bottom)
#             separator_row = 130
            
#             # Store combined version
#             combined_cache[filename] = combined_image
            
#             # Create ECG-only version (white out bottom)
#             ecg_only = combined_image.copy()
#             ecg_only[:, separator_row:, :] = 255
#             ecg_cache[filename] = ecg_only
            
#             # Create VCG-only version (white out top)
#             vcg_only = combined_image.copy()
#             vcg_only[:, :separator_row, :] = 255
#             vcg_cache[filename] = vcg_only
#         else:
#             invalid_paths.append(filename)
            
#     except Exception as e:
#         invalid_paths.append(filename)

# # --- 5. Saving ---

# print(f"--- Processing Complete ---")
# print(f"Successfully generated {len(combined_cache)} image sets.")
# if invalid_paths:
#     print(f"Failed to process {len(invalid_paths)} records.")

# # Save ECG-only cache
# try:
#     ecg_filepath = os.path.join("./", "ecg_only.pkl")
#     print(f"Saving ECG-only cache to: {ecg_filepath}...")
#     with open(ecg_filepath, 'wb') as f:
#         pickle.dump(ecg_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
#     print("✅ ECG-only cache saved successfully.")
# except Exception as e:
#     print(f"Error saving ECG cache: {e}")

# # Save VCG-only cache
# try:
#     vcg_filepath = os.path.join("./", "vcg_only.pkl")
#     print(f"Saving VCG-only cache to: {vcg_filepath}...")
#     with open(vcg_filepath, 'wb') as f:
#         pickle.dump(vcg_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
#     print("✅ VCG-only cache saved successfully.")
# except Exception as e:
#     print(f"Error saving VCG cache: {e}")

# # Save Combined cache
# try:
#     combined_filepath = os.path.join("./", "combined_ecg_vcg.pkl")
#     print(f"Saving combined cache to: {combined_filepath}...")
#     with open(combined_filepath, 'wb') as f:
#         pickle.dump(combined_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
#     print("✅ Combined cache saved successfully.")
# except Exception as e:
#     print(f"Error saving combined cache: {e}")


# --- 4. Parallel Worker Logic ---

def process_record_wrapper(args):
    """
    Worker function to run in a separate process.
    Reads file, generates combined image, splits it into 3 parts.
    Returns: (filename, combined, ecg_only, vcg_only) or (filename, None, None, None)
    """
    filename, folderpath, target_size = args
    full_path = os.path.join(folderpath, filename)
    
    try:
        # Disk I/O
        record = wfdb.rdrecord(full_path)
        
        # CPU Processing
        combined_image = generate_combined_image(record, target_size=target_size)
        
        if combined_image is not None:
            separator_row = 130
            
            # 1. Combined
            # (Already generated)
            
            # 2. ECG Only (White out bottom)
            ecg_only = combined_image.copy()
            ecg_only[:, separator_row:, :] = 255
            
            # 3. VCG Only (White out top)
            vcg_only = combined_image.copy()
            vcg_only[:, :separator_row, :] = 255
            
            return filename, combined_image, ecg_only, vcg_only
        else:
            return filename, None, None, None
            
    except Exception:
        return filename, None, None, None

# --- 5. Main Execution ---

if __name__ == '__main__':
    folderpath = '/Users/alinawaf/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'

    df_full = pd.read_csv(folderpath + "ptbxl_database.csv")
    df_full = df_full.dropna(subset=["age", "filename_hr"])
    df_full = df_full[(df_full["age"] >= 18) & (df_full["age"] <= 80)]

    # Containers for results
    ecg_cache = {}
    vcg_cache = {}
    combined_cache = {}
    invalid_paths = []

    # Configure Parallel Processing
    num_workers = multiprocessing.cpu_count()
    print(f"Starting parallel generation with {num_workers} CPU cores...")

    # Create task list
    # Remove .head(5) to process the full dataset
    tasks = [
        (row["filename_hr"], folderpath, (224, 224)) 
        for _, row in df_full.iterrows()
    ]

    # Execute
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks
        futures = {executor.submit(process_record_wrapper, task): task for task in tasks}
        
        # Gather results with progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
            filename, combined, ecg, vcg = future.result()
            
            if combined is not None:
                combined_cache[filename] = combined
                ecg_cache[filename] = ecg
                vcg_cache[filename] = vcg
            else:
                invalid_paths.append(filename)

    # --- 6. Saving ---

    print(f"--- Processing Complete ---")
    print(f"Successfully generated {len(combined_cache)} image sets.")
    if invalid_paths:
        print(f"Failed to process {len(invalid_paths)} records.")

    # Save ECG-only cache
    try:
        ecg_filepath = os.path.join("./", "ecg_only.pkl")
        print(f"Saving ECG-only cache to: {ecg_filepath}...")
        with open(ecg_filepath, 'wb') as f:
            pickle.dump(ecg_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✅ ECG-only cache saved successfully.")
    except Exception as e:
        print(f"Error saving ECG cache: {e}")

    # Save VCG-only cache
    try:
        vcg_filepath = os.path.join("./", "vcg_only.pkl")
        print(f"Saving VCG-only cache to: {vcg_filepath}...")
        with open(vcg_filepath, 'wb') as f:
            pickle.dump(vcg_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✅ VCG-only cache saved successfully.")
    except Exception as e:
        print(f"Error saving VCG cache: {e}")

    # Save Combined cache
    try:
        combined_filepath = os.path.join("./", "combined_ecg_vcg.pkl")
        print(f"Saving combined cache to: {combined_filepath}...")
        with open(combined_filepath, 'wb') as f:
            pickle.dump(combined_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✅ Combined cache saved successfully.")
    except Exception as e:
        print(f"Error saving combined cache: {e}")