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
    """
    NOTE: This function is now DEPRECATED by the new combined image 
    function, but is kept for reference.
    """
    left = 0.17
    bottom = 8.5 - 7.85
    width = 11 - 2 * left
    height = 7.85 - 2.3
    ax = fig.add_axes([left / 11, bottom / 8.5, width / 11, height / 8.5])

    y_max = height * 25.4 / MM_PER_MV
    ax.set_ylim(0, y_max)
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False, labelbottom=False)

    ax.set_facecolor('white')
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
    
    # Grid lines are removed
    # ax.grid(which='major', color='red', linestyle='-', linewidth=0.4)
    # ax.grid(which='minor', color='red', linestyle=':', linewidth=0.2)

def _plot_ecg_leads(ax: plt.Axes, ecg_data: pd.DataFrame, total_samples: int, show_separators: bool = True):
    leads_layout = [
        ['I',  'AVR', 'V1', 'V4'],
        ['II', 'AVL', 'V2', 'V5'],
        ['III','AVF', 'V3', 'V6']
    ]
    rhythm_strips = ['V1', 'II', 'V5']

    data = ecg_data.iloc[:total_samples].copy()
    total_samples = len(data)

    total_columns = 4
    segment_length = total_samples // total_columns
    boundaries = [i * segment_length for i in range(1, total_columns)]

    total_rows = 6
    y0, y1 = ax.get_ylim()
    y_offset = y1 / total_rows # Use the calculated ylim

    # Plot 12-lead rows
    for row_idx, lead_row in enumerate(leads_layout):
        for col_idx, lead in enumerate(lead_row):
            if lead not in data.columns or data[lead].isnull().all(): continue
            seg_start = col_idx * segment_length
            seg_end = total_samples if col_idx == total_columns - 1 else (col_idx + 1) * segment_length
            offset = (total_rows - row_idx - 0.5) * y_offset
            ax.plot(np.arange(seg_start, seg_end), data[lead].values[seg_start:seg_end] + offset, linewidth=0.6, color='black')

    # Rhythm strips
    for i, lead in enumerate(rhythm_strips):
        if lead not in data.columns or data[lead].isnull().all(): continue
        offset = (total_rows - (3 + i) - 0.5) * y_offset
        ax.plot(np.arange(total_samples), data[lead].values[:total_samples] + offset, linewidth=0.6, color='black')

    if show_separators:
        for x in boundaries:
            ax.axvline(x=x, ymin=0, ymax=1, color='red', linewidth=0.4, alpha=0.6)

def _add_lead_label(ax: plt.Axes, x_pos: float, y_pos: float, text: str):
    # This function is not called but is preserved as requested.
    ax.text(x_pos, y_pos, text, ha='left', va='top', weight='bold')

# --- 2. VCG Helper Functions (Unchanged) ---

def _plot_vcg_loop_simple(ax: plt.Axes, vcg_signals: np.ndarray, projection='frontal'):
    if vcg_signals.shape[1] != 3:
        return 
    
    vx, vy, vz = vcg_signals.T
    
    if projection.lower() == 'frontal' or projection.lower() == 'xy':
        proj_x, proj_y = vx, vy
    elif projection.lower() == 'horizontal' or projection.lower() == 'xz':
        proj_x, proj_y = vx, vz
    elif projection.lower() == 'sagittal' or projection.lower() == 'yz':
        proj_x, proj_y = vy, vz
    else:
        proj_x, proj_y = vx, vy
    
    ax.plot(proj_x, proj_y, 'black', linewidth=1.5, alpha=0.9)
    ax.axis('equal')
    ax.grid(False)
    ax.axis('off')

def get_vcg_signals(record: wfdb.Record) -> np.ndarray:
    """
    Generates VCG signals from a record using CardioVectorLib.
    (This logic is extracted from your original generate_vcg_image)
    """
    try:
        record_copy = record
        record_copy.sig_name = [name.lower() for name in record.sig_name]
        record_copy.base_datetime = None
        
        try:
            filtered_record = preprocessing.remove_baseline_wandering(record_copy)
        except Exception as e:
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
        
        try:
            vcg_record = rec.kors_vcg(filtered_record)
            if vcg_record is None: return None
                
            if not hasattr(vcg_record, 'p_signal') or vcg_record.p_signal is None:
                if hasattr(vcg_record, 'd_signal') and vcg_record.d_signal is not None:
                    vcg_signals = vcg_record.d_signal.astype(float)
                else: return None
            else:
                vcg_signals = vcg_record.p_signal
            
            if vcg_signals.shape[1] != 3: return None
            return vcg_signals
                
        except Exception as e:
            # print(f"❌ CardioVectorLib VCG generation failed: {e}")
            return None
    except Exception as e:
        # print(f"VCG signal generation error: {e}")
        return None

# --- 3. OLD Image Generation Functions (Kept for reference) ---

def generate_and_process_image(record: wfdb.Record, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    DEPRECATED: Generates ONLY the ECG image.
    """
    df = pd.DataFrame(record.p_signal, columns=record.sig_name)
    ecg_data = apply_baseline_correction(df)
    if ecg_data.empty or ecg_data.isnull().all().all():
        return None

    fig = plt.figure(figsize=(11, 8.5), dpi=100)
    ax = _create_ecg_axes(fig) # Uses the old, full-page axes
    total_samples = len(ecg_data)
    _configure_axes_grid(ax, total_samples) 
    _plot_ecg_leads(ax, ecg_data, total_samples, show_separators=False)

    fig.canvas.draw()
    img_buf = fig.canvas.buffer_rgba()
    img_array_rgba = np.asarray(img_buf)
    plt.close(fig) 

    return img_array_rgba # (modified for brevity)


def generate_vcg_image(record: wfdb.Record, target_size: tuple = (224, 224), 
                      projection='frontal', style='single') -> np.ndarray:
    """
    DEPRECATED: Generates ONLY the VCG image(s).
    """
    vcg_signals = get_vcg_signals(record)
    if vcg_signals is None:
        return None

    if style == 'triple':
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
        fig.set_facecolor('white')
        _plot_vcg_loop_simple(ax1, vcg_signals, projection='frontal')
        _plot_vcg_loop_simple(ax2, vcg_signals, projection='horizontal')
        _plot_vcg_loop_simple(ax3, vcg_signals, projection='sagittal')
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
        fig.set_facecolor('white')
        _plot_vcg_loop_simple(ax, vcg_signals, projection=projection)
    
    fig.canvas.draw()
    img_buf = fig.canvas.buffer_rgba()
    img_array_rgba = np.asarray(img_buf)
    plt.close(fig) 
    # ... (rest of processing) ...
    return img_array_rgba # (modified for brevity)


# --- 4. NEW Combined Image Generation Function ---

def generate_combined_ecg_vcg_image(record: wfdb.Record, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Generates a single image with the ECG plot on top
    and the VCG (triple loop) plots in the remaining area below.
    """
    
    # 1. Get ECG Data
    df = pd.DataFrame(record.p_signal, columns=record.sig_name)
    ecg_data = apply_baseline_correction(df)
    if ecg_data.empty or ecg_data.isnull().all().all():
        # print("ECG data is empty")
        return None
    total_samples = len(ecg_data)

    # 2. Get VCG Data
    vcg_signals = get_vcg_signals(record)
    if vcg_signals is None:
        # print("VCG data is None")
        return None

    # 3. Create the master figure
    fig = plt.figure(figsize=(11, 8.5), dpi=300)
    fig.set_facecolor('white') # Ensure background is white


    # [L, B, W, H] percentages

    ecg_ax = fig.add_axes([0.05, 0.25, 0.9, 0.7]) 
    
    # Configure the ECG axes (mimicking _create_ecg_axes)
    # Use the new axes height (0.7 * 8.5 inches) to calculate ylim
    ecg_ax_height_inches = 0.7 * 8.5 
    y_max = ecg_ax_height_inches * 25.4 / MM_PER_MV
    ecg_ax.set_ylim(0, y_max)
    
    ecg_ax.tick_params(which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
    ecg_ax.set_facecolor('white')
    for spine in ecg_ax.spines.values():
        spine.set_visible(False)

    # Plot ECG data onto the axes
    _configure_axes_grid(ecg_ax, total_samples) 
    _plot_ecg_leads(ecg_ax, ecg_data, total_samples, show_separators=False)

    # 5. Define layout and plot VCG (Bottom "remaining" section)
    # We use the bottom ~20% of the figure for the VCGs
    
    # Define 3 axes for the triple-loop
    # [L, B, W, H]
    vcg_ax1 = fig.add_axes([0.05, 0.05, 0.28, 0.15]) # Frontal
    vcg_ax2 = fig.add_axes([0.36, 0.05, 0.28, 0.15]) # Horizontal
    vcg_ax3 = fig.add_axes([0.67, 0.05, 0.28, 0.15]) # Sagittal

    # Plot VCG loops onto their respective axes
    _plot_vcg_loop_simple(vcg_ax1, vcg_signals, projection='frontal')
    _plot_vcg_loop_simple(vcg_ax2, vcg_signals, projection='horizontal')
    _plot_vcg_loop_simple(vcg_ax3, vcg_signals, projection='sagittal')
    
    # 6. Render the entire figure to a NumPy array
    try:
        fig.canvas.draw()
        img_buf = fig.canvas.buffer_rgba()
        img_array_rgba = np.asarray(img_buf)
    except Exception as e:
        print(f"Error during figure rendering: {e}")
        plt.close(fig)
        return None
        
    plt.close(fig) 

    # 7. Post-process the image (convert, resize)
    try:
        img_array_bgr = cv2.cvtColor(img_array_rgba, cv2.COLOR_RGBA2BGR)
        img_array_gray = cv2.cvtColor(img_array_bgr, cv2.COLOR_BGR2GRAY)
        # nearest neighbors
        img_resized = cv2.resize(img_array_gray, target_size, interpolation=cv2.INTER_AREA)
        img_final = np.expand_dims(img_resized, axis=0)
        
        return img_final
    except Exception as e:
        print(f"Error during image post-processing: {e}")
        return None

# --- 5. Main Script (MODIFIED) ---

folderpath = '/Users/alinawaf/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'

# --- NEW CACHE FILENAME ---
cache_filename = "combined_ecg_vcg_image_cache_hr_1x224x224.pkl" 
cache_filepath = os.path.join("./", cache_filename)

df_full = pd.read_csv(folderpath + "ptbxl_database.csv")
df_full = df_full.dropna(subset=["age", "filename_hr"])
df_full = df_full[(df_full["age"] >= 18) & (df_full["age"] <= 80)]

# --- MODIFIED CACHE STRUCTURE ---
# The cache will now store: { filename: combined_image_array }
ecg_vcg_cache = {}
invalid_paths = []

print("Starting to generate and cache COMBINED ECG+VCG (1, 224, 224) images...")

# --- MODIFIED Caching Loop ---
for _, row in tqdm(df_full.iterrows(), total= len(df_full) ):
    filename = row["filename_hr"] 
    full_path = os.path.join(folderpath, filename)
    
    try:
        record = wfdb.rdrecord(full_path)
        
        # --- SINGLE CALL to the new combined function ---
        combined_image = generate_combined_ecg_vcg_image(record)
        
        if combined_image is not None:
            ecg_vcg_cache[filename] = combined_image
        else:
            invalid_paths.append(filename)
            
    except Exception as e:
        # print(f"Warning: Could not load {full_path}. Error: {e}")
        invalid_paths.append(filename)

# --- 6. Save the New Image Cache ---

print(f"--- Caching Complete ---")
print(f"Successfully generated and cached {len(ecg_vcg_cache)} combined ECG-VCG images.")
if invalid_paths:
    print(f"Failed to process {len(invalid_paths)} records.")

try:
    print(f"Saving COMBINED image cache to: {cache_filepath}...")
    with open(cache_filepath, 'wb') as f:
        pickle.dump(ecg_vcg_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("✅ Combined image cache saved.")
except Exception as e:
    print(f"Error saving cache: {e}")