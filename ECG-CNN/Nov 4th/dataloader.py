import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import wfdb
import matplotlib
import matplotlib.pyplot as plt
import cv2  # Make sure cv2 is imported
import pickle
from scipy.signal import butter, filtfilt
import sys
sys.path.append('/Users/alinawaf/Desktop/Research/ECG-VECG/CardioVectorLib')
from cardiovector import plotting, preprocessing, reconstruction as rec


# --- Matplotlib Configuration ---
matplotlib.use('Agg') # Use 'Agg' backend for non-GUI rendering
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9.5

# --- Constants ---
MM_PER_MV = 10
MM_PER_SEC = 25
SAMPLING_RATE = 500  # Hz (for 'filename_hr' data)

# --- 1. Preprocessing & Plotting Helper Functions ---

# Kept your more robust baseline correction from the caching script
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

# --- NEW Plotting Helpers (from your sample script) ---

def _add_demographic_text(fig: plt.Figure):
    # This function is not called but is preserved as requested.
    demo_text = [
        ("Patient: DOE, JOHN", 0.17, 8.04),
        ("ID: 123456789", 3.05, 8.04),
        # ... (other text entries) ...
        (f"{MM_PER_SEC}mm/s    {MM_PER_MV}mm/mV    {SAMPLING_RATE}Hz", 0.17, 0.46)
    ]
    for text, x, y in demo_text:
        fig.text(x / 11, y / 8.5, text, weight='bold')

def _create_ecg_axes(fig: plt.Figure) -> plt.Axes:
    left = 0.17
    bottom = 8.5 - 7.85
    width = 11 - 2 * left
    height = 7.85 - 2.3
    ax = fig.add_axes([left / 11, bottom / 8.5, width / 11, height / 8.5])

    y_max = height * 25.4 / MM_PER_MV
    ax.set_ylim(0, y_max)
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False, labelbottom=False)

    # --- CHANGES APPLIED ---
    # Explicitly set the background to white
    ax.set_facecolor('white')
    # Remove the boundary box/frame around the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    return ax

def _configure_axes_grid(ax: plt.Axes, total_samples: int):
    ax.set_xlim(0, total_samples)

    samples_per_mm = SAMPLING_RATE / MM_PER_SEC
    small_x_step = samples_per_mm
    big_x_step = small_x_step * 5

    small_y_step = 0.1
    big_y_step = 0.5

    ax.set_xticks(np.arange(0, total_samples + 1, big_x_step))
    ax.set_xticks(np.arange(0, total_samples + 1, small_x_step), minor=True)
    y0, y1 = ax.get_ylim()
    ax.set_yticks(np.arange(0, y1 + 1e-6, big_y_step))
    ax.set_yticks(np.arange(0, y1 + 1e-6, small_y_step), minor=True)

    # --- CHANGES APPLIED ---
    # Grid lines are commented out to remove them
    # ax.grid(which='major', color='red', linestyle='-', linewidth=0.4)
    # ax.grid(which='minor', color='red', linestyle=':', linewidth=0.2)

def _plot_ecg_leads(ax: plt.Axes, ecg_data: pd.DataFrame, total_samples: int, show_separators: bool = True):
    leads_layout = [ # BIG PROBLEM HERE CAUSE ALOT OF ISSUES
        ['I',  'AVR', 'V1', 'V4'],  # Changed aVR -> AVR
        ['II', 'AVL', 'V2', 'V5'],  # Changed aVL -> AVL
        ['III','AVF', 'V3', 'V6']   # Changed aVF -> AVF
    ]
    # rhythm_strips = ['V1', 'II', 'V5']

    data = ecg_data.iloc[:total_samples].copy()
    total_samples = len(data)

    total_columns = 4
    segment_length = total_samples // total_columns
    boundaries = [i * segment_length for i in range(1, total_columns)]

    total_rows = 3
    y_offset = ax.get_ylim()[1] / total_rows

    # Plot 12-lead rows
    for row_idx, lead_row in enumerate(leads_layout):
        for col_idx, lead in enumerate(lead_row):
            if lead not in data.columns or data[lead].isnull().all(): continue
            seg_start = col_idx * segment_length
            seg_end = total_samples if col_idx == total_columns - 1 else (col_idx + 1) * segment_length
            offset = (total_rows - row_idx - 0.5) * y_offset
            ax.plot(np.arange(seg_start, seg_end), data[lead].values[seg_start:seg_end] + offset, linewidth=0.6, color='black')
            # --- CHANGES APPLIED ---
            # _add_lead_label(ax, seg_start + 5, offset - 0.25, lead)

    # # Rhythm strips
    # for i, lead in enumerate(rhythm_strips):
    #     if lead not in data.columns or data[lead].isnull().all(): continue
    #     offset = (total_rows - (3 + i) - 0.5) * y_offset
    #     ax.plot(np.arange(total_samples), data[lead].values[:total_samples] + offset, linewidth=0.6, color='black')
        # --- CHANGES APPLIED ---
        # _add_lead_label(ax, 5, offset - 0.25, lead)

    # This is also commented out in your sample script's main call (show_separators=False)
    if show_separators:
        for x in boundaries:
            ax.axvline(x=x, ymin=0, ymax=1, color='red', linewidth=0.4, alpha=0.6)

###  VCG drawing and extraction 
###  VCG drawing and extraction 

def _plot_vcg_loop_simple(ax: plt.Axes, vcg_signals: np.ndarray, projection='frontal'):
    """
    Plots a single, clean VCG projection onto a given axes using the
    correct method (ax.plot, ax.axis('equal')).
    """
    if vcg_signals.shape[1] != 3:
        return 
    
    vx, vy, vz = vcg_signals.T
    
    # Select the correct projection
    if projection.lower() == 'frontal' or projection.lower() == 'xy':
        proj_x, proj_y = vx, vy
    elif projection.lower() == 'horizontal' or projection.lower() == 'xz':
        proj_x, proj_y = vx, vz
    elif projection.lower() == 'sagittal' or projection.lower() == 'yz':
        proj_x, proj_y = vy, vz
    else:
        proj_x, proj_y = vx, vy  # Default to frontal
    
    # Plot the loop
    ax.plot(proj_x, proj_y, 'black', linewidth=1.5, alpha=0.9)
    
    # --- Critical Fixes from your "good" script ---
    
    # 1. Use 'equal' axis scaling to prevent distortion
    ax.axis('equal')
    
    # 2. Turn off all grids, labels, and ticks for a clean image
    ax.grid(False)
    ax.axis('off')

# NOTE: The old _plot_vcg_triple_loops function is no longer needed
# and can be deleted. The logic is now inside generate_vcg_image.

def generate_vcg_image(record: wfdb.Record, target_size: tuple = (224, 224), 
                      projection='frontal', style='single') -> np.ndarray:
    """
    Generate VCG loop image using CardioVectorLib approach.
    This version correctly handles 'single' vs 'triple' plot styles.
    """
    try:
        # --- (Your existing VCG signal generation code) ---
        # --- (This part is unchanged) ---
        
        # Prepare record for CardioVectorLib
        record_copy = record
        record_copy.sig_name = [name.lower() for name in record.sig_name]
        record_copy.base_datetime = None
        
        # Apply baseline filtering using CardioVectorLib
        try:
            filtered_record = preprocessing.remove_baseline_wandering(record_copy)
        except Exception as e:
            # Fallback to manual filtering
            df = pd.DataFrame(record.p_signal, columns=record.sig_name)
            nyquist = 0.5 * 500
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
        
        # Generate VCG using CardioVectorLib Kors transformation
        try:
            vcg_record = rec.kors_vcg(filtered_record)
            
            if vcg_record is None:
                print("❌ VCG record is None - CardioVectorLib KORS transform failed")
                return None
                
            if not hasattr(vcg_record, 'p_signal') or vcg_record.p_signal is None:
                if hasattr(vcg_record, 'd_signal') and vcg_record.d_signal is not None:
                    vcg_signals = vcg_record.d_signal.astype(float)
                else:
                    print("❌ VCG record has no signal data")
                    return None
            else:
                vcg_signals = vcg_record.p_signal
            
            if vcg_signals.shape[1] != 3:
                print(f"❌ Expected 3 VCG components, got {vcg_signals.shape[1]}")
                return None
                
        except Exception as e:
            print(f"❌ CardioVectorLib VCG generation failed: {e}")
            return None
        
        # --- (End of your existing signal generation code) ---


        # --- NEW PLOTTING SECTION ---
        # This section creates the correct figure and axes
        # instead of using _create_ecg_axes
        
        if style == 'triple':
            # Create a 1x3 grid of subplots.
            # A 12x4 figsize gives a 3:1 aspect ratio, good for 3 plots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
            fig.set_facecolor('white') # Ensure background is white

            # Plot each projection on its own dedicated axes
            _plot_vcg_loop_simple(ax1, vcg_signals, projection='frontal')
            _plot_vcg_loop_simple(ax2, vcg_signals, projection='horizontal')
            _plot_vcg_loop_simple(ax3, vcg_signals, projection='sagittal')
            
            plt.tight_layout() # Adjust subplots to fit
        
        else: # style == 'single'
            # Create a single plot with a 1:1 aspect ratio
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
            fig.set_facecolor('white')
            
            # Plot the single requested projection
            _plot_vcg_loop_simple(ax, vcg_signals, projection=projection)
        
        # --- END NEW PLOTTING SECTION ---

        
        # Render to image (This part is unchanged)
        fig.canvas.draw()
        img_buf = fig.canvas.buffer_rgba()
        img_array_rgba = np.asarray(img_buf)
        plt.close(fig) 
        
        # Convert to grayscale and resize (This part is unchanged)
        img_array_bgr = cv2.cvtColor(img_array_rgba, cv2.COLOR_RGBA2BGR)
        img_array_gray = cv2.cvtColor(img_array_bgr, cv2.COLOR_BGR2GRAY)
        
        img_resized = cv2.resize(img_array_gray, target_size, interpolation=cv2.INTER_AREA)
        img_final = np.expand_dims(img_resized, axis=0)
        
        return img_final
        
    except Exception as e:
        print(f"VCG generation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def _add_lead_label(ax: plt.Axes, x_pos: float, y_pos: float, text: str):
    # This function is not called but is preserved as requested.
    ax.text(x_pos, y_pos, text, ha='left', va='top', weight='bold')

# --- 2. Master Image Generation Function ---
# (This function is unchanged, as it just calls the helper
# functions which we have now replaced)

def generate_and_process_image(record: wfdb.Record, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Plots, renders, converts to B&W, resizes, and reshapes
    the ECG record into a (1, 224, 224) array.
    """
    df = pd.DataFrame(record.p_signal, columns=record.sig_name)
    ecg_data = apply_baseline_correction(df)

    if ecg_data.empty or ecg_data.isnull().all().all():
        return None

    # This will now use the new "clean" plotting functions
    fig = plt.figure(figsize=(11, 8.5), dpi=100)
    ax = _create_ecg_axes(fig)
    total_samples = len(ecg_data)
    _configure_axes_grid(ax, total_samples) 
    _plot_ecg_leads(ax, ecg_data, total_samples, show_separators=False)

    fig.canvas.draw()
    img_buf = fig.canvas.buffer_rgba()
    img_array_rgba = np.asarray(img_buf)
    plt.close(fig) 

    img_array_bgr = cv2.cvtColor(img_array_rgba, cv2.COLOR_RGBA2BGR)
    img_array_gray = cv2.cvtColor(img_array_bgr, cv2.COLOR_BGR2GRAY)
    
    img_resized = cv2.resize(img_array_gray, target_size, interpolation=cv2.INTER_AREA)
    img_final = np.expand_dims(img_resized, axis=0)

    return img_final

# --- 3. Main Script (Unchanged) ---

# NOTE: Mount drive if you are on Google Colab
# from google.colab import drive
# drive.mount("/content/drive")
# folderpath = '/content/drive/MyDrive/CNN/'

# Using the local path from your script
folderpath = '/Users/alinawaf/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'

cache_filename = "ecg_vcg_image_cache_hr_1x224x224.pkl" 
cache_filepath = os.path.join("./", cache_filename)

df_full = pd.read_csv(folderpath + "ptbxl_database.csv")

df_full = df_full.dropna(subset=["age", "filename_hr"])
df_full = df_full[(df_full["age"] >= 18) & (df_full["age"] <= 80)]

ecg_vcg_cache = {}
invalid_paths = []

print("Starting to generate and cache ECG and VCG (1, 224, 224) images from 500 Hz data...")

# Update the caching loop
for _, row in tqdm(df_full.iterrows(), total=len(df_full)):
# for _, row in tqdm(df_full.head(5).iterrows(), total = 5):
    filename = row["filename_hr"] 
    full_path = os.path.join(folderpath, filename)
    
    try:
        record = wfdb.rdrecord(full_path)
        
        # Generate ECG image
        ecg_image = generate_and_process_image(record)
        
        # Generate VCG loop image using CardioVectorLib
        # Choose your preferred style:
        # vcg_image = generate_vcg_image(record, projection='frontal', style='single')  # Single frontal loop
        vcg_image = generate_vcg_image(record, style='triple')  # All three projections
        
        # Store both if successful
        if ecg_image is not None and vcg_image is not None:
            ecg_vcg_cache[filename] = {
                'ecg': ecg_image,
                'vcg': vcg_image
            }
        else:
            invalid_paths.append(filename)
            
    except Exception as e:
        # print(f"Warning: Could not load {full_path}. Error: {e}")
        invalid_paths.append(filename)

# --- 4. Save the New Image Cache ---

print(f"--- Caching Complete ---")
print(f"Successfully generated and cached {len(ecg_vcg_cache)} ECG-VCG image pairs.")
if invalid_paths:
    print(f"Failed to process {len(invalid_paths)} records.")

try:
    print(f"Saving ECG-VCG image cache to: {cache_filepath}...")
    with open(cache_filepath, 'wb') as f:
        pickle.dump(ecg_vcg_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("✅ ECG-VCG image cache saved.")
except Exception as e:
    print(f"Error saving cache: {e}")