import wfdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from IPython.display import Image, display
import matplotlib
from scipy.signal import butter, filtfilt

# Matplotlib config
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9.5

# Constants
MM_PER_MV = 10
MM_PER_SEC = 25
SAMPLING_RATE = 500  # Hz

# === Preprocessing ===
def read_wfdb_to_csv(record_path: Path, output_dir: Path) -> Path:
    record = wfdb.rdrecord(str(record_path))
    df = pd.DataFrame(record.p_signal, columns=record.sig_name)
    csv_path = output_dir / f"{record_path.stem}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    return csv_path

def load_ecg_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=None, engine='python')
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(how='all').dropna(axis=1, how='all')

def apply_baseline_correction(df: pd.DataFrame, sampling_rate: float = SAMPLING_RATE, cutoff_hz: float = 0.67) -> pd.DataFrame:
    """High-pass filter to remove baseline wander."""
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(N=2, Wn=normal_cutoff, btype='high', analog=False)

    filtered_df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            filtered_df[col] = filtfilt(b, a, df[col].values)
    return filtered_df

# === Plotting ===
def plot_ecg_from_csv(csv_path: Path, output_path: Path, show_grid: bool = True):
    ecg_data = apply_baseline_correction(load_ecg_csv(csv_path))
    if ecg_data.empty:
        print("No valid ECG data loaded")
        return

    # Get available leads (columns)
    available_leads = [col for col in ecg_data.columns if pd.api.types.is_numeric_dtype(ecg_data[col])]
    num_leads = len(available_leads)
    
    if num_leads == 0:
        print("No numeric lead data found")
        return

    # Calculate figure height based on number of leads
    fig_height = max(8.5, num_leads * 1.2)  # Minimum height, scale with leads
    fig = plt.figure(figsize=(11, fig_height), dpi=100)
    
    _add_demographic_text(fig, fig_height)
    ax = _create_ecg_axes(fig, fig_height, num_leads)

    max_samples = max(len(ecg_data[lead].dropna()) for lead in available_leads)
    _configure_axes_grid(ax, max_samples, num_leads, show_grid)
    _plot_leads_sequentially(ax, ecg_data, available_leads, num_leads)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"ECG plot saved to {output_path}")

def _add_demographic_text(fig: plt.Figure, fig_height: float):
    # Scale positions based on figure height
    scale_factor = fig_height / 8.5
    
    demo_text = [
        ("Patient: DOE, JOHN", 0.17, fig_height - 0.46),
        ("ID: 123456789", 3.05, fig_height - 0.46),
        ("01-JAN-2023 10:30:45", 4.56, fig_height - 0.46),
        ("Hospital ECG Lab", 6.05, fig_height - 0.46),
        ("DOB: 01-JAN-1970 (53 yr)", 0.17, fig_height - 0.73),
        ("Male", 0.17, fig_height - 0.87),
        ("Vent. rate: 72", 2.15, fig_height - 0.73),
        ("PR interval: 160", 2.15, fig_height - 0.87),
        ("QRS duration: 88", 2.15, fig_height - 1.01),
        ("QT/QTc: 400/420", 2.15, fig_height - 1.15),
        ("P-R-T axes: 45 60 30", 2.15, fig_height - 1.29),
        (f"{MM_PER_SEC}mm/s    {MM_PER_MV}mm/mV    {SAMPLING_RATE}Hz", 0.17, 0.46)
    ]
    for text, x, y in demo_text:
        fig.text(x / 11, y / fig_height, text, weight='bold')

def _create_ecg_axes(fig: plt.Figure, fig_height: float, num_leads: int) -> plt.Axes:
    left = 0.17
    bottom = 0.8  # Fixed bottom margin
    width = 11 - 2 * left
    height = fig_height - 2.3  # Leave space for header and footer
    
    ax = fig.add_axes([left / 11, bottom / fig_height, width / 11, height / fig_height])

    # Set y-limits based on number of leads with some padding between leads
    y_max = num_leads * 2.0  # 2.0 units per lead for good separation
    ax.set_ylim(0, y_max)
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
    return ax

def _configure_axes_grid(ax: plt.Axes, max_samples: int, num_leads: int, show_grid: bool):
    ax.set_xlim(0, max_samples)

    if show_grid:
        samples_per_mm = SAMPLING_RATE / MM_PER_SEC  # 20 samples/mm
        small_x_step = samples_per_mm
        big_x_step = small_x_step * 5

        # Scale grid spacing for voltage based on lead separation
        voltage_per_lead = 2.0  # units per lead
        small_y_step = 0.1 * voltage_per_lead
        big_y_step = 0.5 * voltage_per_lead

        ax.set_xticks(np.arange(0, max_samples + 1, big_x_step))
        ax.set_xticks(np.arange(0, max_samples + 1, small_x_step), minor=True)
        y0, y1 = ax.get_ylim()
        ax.set_yticks(np.arange(0, y1 + 1e-6, big_y_step))
        ax.set_yticks(np.arange(0, y1 + 1e-6, small_y_step), minor=True)

        ax.grid(which='major', color='red', linestyle='-', linewidth=0.4)
        ax.grid(which='minor', color='red', linestyle=':', linewidth=0.2)

def _plot_leads_sequentially(ax: plt.Axes, ecg_data: pd.DataFrame, available_leads: list, num_leads: int):
    """Plot each lead as a separate row, regardless of traditional 12-lead layout."""
    
    voltage_per_lead = 2.0  # Vertical spacing between leads
    
    for i, lead in enumerate(available_leads):
        # Calculate vertical offset for this lead (from bottom to top)
        y_offset = (num_leads - i - 0.5) * voltage_per_lead
        
        # Get the data for this lead, removing NaN values
        lead_data = ecg_data[lead].dropna().values
        
        if len(lead_data) == 0:
            continue
            
        # Create time axis for this lead
        x_values = np.arange(len(lead_data))
        
        # Normalize the signal amplitude to fit nicely in the allocated space
        # Scale to use about 80% of the available vertical space for this lead
        signal_scale = 0.8 * voltage_per_lead
        normalized_signal = (lead_data - np.mean(lead_data)) / (np.std(lead_data) + 1e-8) * signal_scale * 0.2
        
        # Plot the lead
        ax.plot(x_values, normalized_signal + y_offset, 
               linewidth=0.8, color='black', label=lead)
        
        # Add lead label
        _add_lead_label(ax, 50, y_offset + 0.3, lead)
        
        # Add a subtle horizontal line to separate leads
        ax.axhline(y=y_offset - voltage_per_lead/2, color='lightgray', 
                  linestyle='--', linewidth=0.3, alpha=0.5)

def _add_lead_label(ax: plt.Axes, x_pos: float, y_pos: float, text: str):
    ax.text(x_pos, y_pos, text, ha='left', va='center', weight='bold', 
            fontsize=10, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# ==== MAIN ====
if __name__ == '__main__':
    record_path = Path('/Users/alinawaf/Desktop/Research/ECG-VECG/Output/1/00001_-0_0000')
    output_dir = Path('/Users/alinawaf/Desktop/Research/ECG-VECG')
    output_img = output_dir / f"{record_path.stem}.png"

    csv_file = read_wfdb_to_csv(record_path, output_dir)
    plot_ecg_from_csv(csv_file, output_img, show_grid=True)
    try:
        display(Image(filename=str(output_img)))
    except Exception:
        pass