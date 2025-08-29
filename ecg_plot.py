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
def plot_ecg_from_csv(csv_path: Path, output_path: Path, show_separators: bool = True):
    ecg_data = apply_baseline_correction(load_ecg_csv(csv_path))
    if ecg_data.empty:
        print("No valid ECG data loaded")
        return

    fig = plt.figure(figsize=(11, 8.5), dpi=100)
    _add_demographic_text(fig)
    ax = _create_ecg_axes(fig)

    total_samples = len(ecg_data)
    _configure_axes_grid(ax, total_samples)
    _plot_ecg_leads(ax, ecg_data, total_samples, show_separators=show_separators)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"ECG plot saved to {output_path}")

def _add_demographic_text(fig: plt.Figure):
    demo_text = [
        ("Patient: DOE, JOHN", 0.17, 8.04),
        ("ID: 123456789", 3.05, 8.04),
        ("01-JAN-2023 10:30:45", 4.56, 8.04),
        ("Hospital ECG Lab", 6.05, 8.04),
        ("DOB: 01-JAN-1970 (53 yr)", 0.17, 7.77),
        ("Male", 0.17, 7.63),
        ("Vent. rate: 72", 2.15, 7.77),
        ("PR interval: 160", 2.15, 7.63),
        ("QRS duration: 88", 2.15, 7.49),
        ("QT/QTc: 400/420", 2.15, 7.35),
        ("P-R-T axes: 45 60 30", 2.15, 7.21),
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
    return ax

def _configure_axes_grid(ax: plt.Axes, total_samples: int):
    ax.set_xlim(0, total_samples)

    samples_per_mm = SAMPLING_RATE / MM_PER_SEC  # 20 samples/mm
    small_x_step = samples_per_mm
    big_x_step = small_x_step * 5

    small_y_step = 0.1
    big_y_step = 0.5

    ax.set_xticks(np.arange(0, total_samples + 1, big_x_step))
    ax.set_xticks(np.arange(0, total_samples + 1, small_x_step), minor=True)
    y0, y1 = ax.get_ylim()
    ax.set_yticks(np.arange(0, y1 + 1e-6, big_y_step))
    ax.set_yticks(np.arange(0, y1 + 1e-6, small_y_step), minor=True)

    ax.grid(which='major', color='red', linestyle='-', linewidth=0.4)
    ax.grid(which='minor', color='red', linestyle=':', linewidth=0.2)

def _plot_ecg_leads(ax: plt.Axes, ecg_data: pd.DataFrame, total_samples: int, show_separators: bool = True):
    leads_layout = [
        ['I',  'aVR', 'V1', 'V4'],
        ['II', 'aVL', 'V2', 'V5'],
        ['III','aVF', 'V3', 'V6']
    ]
    rhythm_strips = ['V1', 'II', 'V5']

    data = ecg_data.iloc[:total_samples].copy()
    total_samples = len(data)

    total_columns = 4
    segment_length = total_samples // total_columns
    boundaries = [i * segment_length for i in range(1, total_columns)]

    total_rows = 6
    y_offset = ax.get_ylim()[1] / total_rows

    # Plot 12-lead rows
    for row_idx, lead_row in enumerate(leads_layout):
        for col_idx, lead in enumerate(lead_row):
            if lead not in data.columns:
                continue
            seg_start = col_idx * segment_length
            seg_end = total_samples if col_idx == total_columns - 1 else (col_idx + 1) * segment_length
            offset = (total_rows - row_idx - 0.5) * y_offset
            ax.plot(np.arange(seg_start, seg_end), data[lead].values[seg_start:seg_end] + offset, linewidth=0.6, color='black')
            _add_lead_label(ax, seg_start + 5, offset - 0.25, lead)  # lowered by 0.25

    # Rhythm strips
    for i, lead in enumerate(rhythm_strips):
        if lead in data.columns:
            offset = (total_rows - (3 + i) - 0.5) * y_offset
            ax.plot(np.arange(total_samples), data[lead].values[:total_samples] + offset, linewidth=0.6, color='black')
            _add_lead_label(ax, 5, offset - 0.25, lead)  # lowered by 0.25

    # Optional separators
    if show_separators:
        for x in boundaries:
            ax.axvline(x=x, ymin=0, ymax=1, color='red', linewidth=0.4, alpha=0.6)

def _add_lead_label(ax: plt.Axes, x_pos: float, y_pos: float, text: str):
    ax.text(x_pos, y_pos, text, ha='left', va='top', weight='bold')

# ==== MAIN ====
if __name__ == '__main__':
    record_path = Path('/Users/alinawaf/Desktop/Research/ECG-VECG/MIMIC_Dataset/s40689238/40689238')
    output_dir = Path('/Users/alinawaf/Desktop/Research/ECG-VECG')
    output_img = output_dir / f"{record_path.stem}.png"

    csv_file = read_wfdb_to_csv(record_path, output_dir)
    plot_ecg_from_csv(csv_file, output_img, show_separators=False)
    try:
        display(Image(filename=str(output_img)))
    except Exception:
        pass
