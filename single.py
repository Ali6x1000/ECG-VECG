import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import sys
import os
from scipy.signal import find_peaks

sys.path.append('/Users/alinawaf/Desktop/Research/ECG-VECG/CardioVectorLib')
from cardiovector import plotting, preprocessing, reconstruction as rec

def create_record_copy(original_record):
    """Create a copy of WFDB record since it doesn't have copy() method"""
    return wfdb.Record(
        record_name=getattr(original_record, 'record_name', 'copied_record'),
        n_sig=original_record.n_sig,
        fs=original_record.fs,
        sig_len=original_record.sig_len,
        p_signal=original_record.p_signal.copy(),
        sig_name=original_record.sig_name.copy(),
        units=getattr(original_record, 'units', ['mV'] * original_record.n_sig),
        comments=getattr(original_record, 'comments', [])
    )

def detect_r_peaks(ecg_signal, fs, min_distance=None):
    """Detect R-peaks in ECG signal"""
    if min_distance is None:
        min_distance = int(0.3 * fs)  # Minimum 300ms between R-peaks
    
    # Use Lead II for R-peak detection (usually best for this)
    if ecg_signal.ndim > 1:
        lead_ii = ecg_signal[:, 1]  # Lead II is index 1
    else:
        lead_ii = ecg_signal
    
    # Find peaks above 75th percentile with minimum distance
    threshold = np.percentile(lead_ii, 75)
    peaks, properties = find_peaks(lead_ii, 
                                  height=threshold,
                                  distance=min_distance,
                                  prominence=0.1)
    
    return peaks

def extract_single_heartbeat(vcg_record, beat_number=0, padding_ms=100):
    """
    Extract a single heartbeat from VCG data
    
    Parameters:
    - vcg_record: WFDB record with VCG data
    - beat_number: which beat to extract (0 = first, 1 = second, etc.)
    - padding_ms: milliseconds to include before and after the beat
    """
    
    # First, we need the original ECG to detect R-peaks
    # Load original ECG data
    record_name = "/Users/alinawaf/Desktop/Research/ECG-VECG/MIMIC_Dataset/s40689238/40689238"
    original_record = wfdb.rdrecord(record_name)
    
    # Detect R-peaks in original ECG
    r_peaks = detect_r_peaks(original_record.p_signal, original_record.fs)
    
    print(f"Detected {len(r_peaks)} R-peaks")
    print(f"R-peak locations (samples): {r_peaks[:10]}")  # Show first 10
    
    if len(r_peaks) <= beat_number:
        print(f"Only {len(r_peaks)} beats found, cannot extract beat {beat_number}")
        return None
    
    # Calculate beat boundaries
    fs = vcg_record.fs
    padding_samples = int(padding_ms * fs / 1000)
    
    if beat_number == 0:
        # First beat: from start to midpoint between first and second R-peak
        start_sample = max(0, r_peaks[0] - padding_samples)
        if len(r_peaks) > 1:
            end_sample = min(len(vcg_record.p_signal), 
                           r_peaks[0] + (r_peaks[1] - r_peaks[0]) // 2 + padding_samples)
        else:
            # Only one beat - use fixed duration
            end_sample = min(len(vcg_record.p_signal), r_peaks[0] + int(0.8 * fs))
    
    elif beat_number == len(r_peaks) - 1:
        # Last beat: from midpoint before last R-peak to end
        start_sample = max(0, r_peaks[-1] - (r_peaks[-1] - r_peaks[-2]) // 2 - padding_samples)
        end_sample = min(len(vcg_record.p_signal), r_peaks[-1] + padding_samples + int(0.4 * fs))
    
    else:
        # Middle beat: from midpoint before to midpoint after
        prev_r = r_peaks[beat_number - 1]
        curr_r = r_peaks[beat_number]
        next_r = r_peaks[beat_number + 1]
        
        start_sample = max(0, curr_r - (curr_r - prev_r) // 2)
        end_sample = min(len(vcg_record.p_signal), curr_r + (next_r - curr_r) // 2)
    
    print(f"Extracting beat {beat_number}: samples {start_sample} to {end_sample}")
    print(f"Duration: {(end_sample - start_sample) / fs:.3f} seconds")
    
    # Extract the single beat VCG data
    single_beat_vcg = vcg_record.p_signal[start_sample:end_sample, :]
    
    # Create new record for the single beat
    beat_record = wfdb.Record(
        record_name=f'single_beat_{beat_number}',
        n_sig=3,
        fs=vcg_record.fs,
        sig_len=len(single_beat_vcg),
        p_signal=single_beat_vcg,
        sig_name=['vx', 'vy', 'vz'],
        units=['mV', 'mV', 'mV'],
        comments=[f'Single heartbeat {beat_number}']
    )
    
    return beat_record, r_peaks[beat_number] - start_sample  # Return R-peak location within beat

def plot_single_beat_vcg(beat_record, r_peak_location=None):
    """Plot single heartbeat VCG with detailed analysis"""
    
    if beat_record is None:
        print("No beat record to plot")
        return
    
    vx, vy, vz = beat_record.p_signal[:, 0], beat_record.p_signal[:, 1], beat_record.p_signal[:, 2]
    time_axis = np.arange(len(vx)) / beat_record.fs
    
    fig = plt.figure(figsize=(16, 12))
    
    # Time domain plot with R-peak marked
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(time_axis, vx, 'r-', label='VX', linewidth=2)
    ax1.plot(time_axis, vy, 'g-', label='VY', linewidth=2)
    ax1.plot(time_axis, vz, 'b-', label='VZ', linewidth=2)
    
    if r_peak_location is not None:
        r_time = r_peak_location / beat_record.fs
        ax1.axvline(x=r_time, color='black', linestyle='--', linewidth=2, label='R-peak')
    
    ax1.set_title('Single Heartbeat - Time Domain')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (mV)')
    ax1.legend()
    ax1.grid(True)
    
    # 3D VCG loop - full beat
    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    colors = np.linspace(0, 1, len(vx))
    scatter = ax2.scatter(vx, vy, vz, c=colors, cmap='plasma', s=20, alpha=0.8)
    ax2.plot(vx, vy, vz, 'k-', alpha=0.6, linewidth=1.5)
    
    # Mark start and end
    ax2.scatter(vx[0], vy[0], vz[0], color='green', s=80, label='Start')
    ax2.scatter(vx[-1], vy[-1], vz[-1], color='red', s=80, label='End')
    
    # Mark R-peak if available
    if r_peak_location is not None and r_peak_location < len(vx):
        ax2.scatter(vx[r_peak_location], vy[r_peak_location], vz[r_peak_location], 
                   color='black', s=100, marker='*', label='R-peak')
    
    ax2.set_xlabel('VX (Left-Right)')
    ax2.set_ylabel('VY (Superior-Inferior)')
    ax2.set_zlabel('VZ (Anterior-Posterior)')
    ax2.set_title('3D VCG - Single Beat')
    ax2.legend()
    
    # Frontal plane (XY)
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(vx, vy, 'b-', linewidth=2, alpha=0.8)
    ax3.scatter(vx[0], vy[0], color='green', s=60, label='Start', zorder=5)
    ax3.scatter(vx[-1], vy[-1], color='red', s=60, label='End', zorder=5)
    
    if r_peak_location is not None and r_peak_location < len(vx):
        ax3.scatter(vx[r_peak_location], vy[r_peak_location], 
                   color='black', s=80, marker='*', label='R-peak', zorder=5)
    
    ax3.set_xlabel('VX (Left-Right)')
    ax3.set_ylabel('VY (Superior-Inferior)')
    ax3.set_title('Frontal Plane (XY)')
    ax3.grid(True)
    ax3.legend()
    ax3.axis('equal')
    
    # Horizontal plane (XZ)
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(vx, vz, 'g-', linewidth=2, alpha=0.8)
    ax4.scatter(vx[0], vz[0], color='green', s=60, label='Start', zorder=5)
    ax4.scatter(vx[-1], vz[-1], color='red', s=60, label='End', zorder=5)
    
    if r_peak_location is not None and r_peak_location < len(vx):
        ax4.scatter(vx[r_peak_location], vz[r_peak_location], 
                   color='black', s=80, marker='*', label='R-peak', zorder=5)
    
    ax4.set_xlabel('VX (Left-Right)')
    ax4.set_ylabel('VZ (Anterior-Posterior)')
    ax4.set_title('Horizontal Plane (XZ)')
    ax4.grid(True)
    ax4.legend()
    ax4.axis('equal')
    
    # Sagittal plane (YZ)
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(vy, vz, 'r-', linewidth=2, alpha=0.8)
    ax5.scatter(vy[0], vz[0], color='green', s=60, label='Start', zorder=5)
    ax5.scatter(vy[-1], vz[-1], color='red', s=60, label='End', zorder=5)
    
    if r_peak_location is not None and r_peak_location < len(vy):
        ax5.scatter(vy[r_peak_location], vz[r_peak_location], 
                   color='black', s=80, marker='*', label='R-peak', zorder=5)
    
    ax5.set_xlabel('VY (Superior-Inferior)')
    ax5.set_ylabel('VZ (Anterior-Posterior)')
    ax5.set_title('Sagittal Plane (YZ)')
    ax5.grid(True)
    ax5.legend()
    ax5.axis('equal')
    
    # Phase portrait (VX vs VY velocity)
    ax6 = fig.add_subplot(3, 3, 6)
    if len(vx) > 1:
        vx_vel = np.gradient(vx)
        vy_vel = np.gradient(vy)
        ax6.plot(vx_vel, vy_vel, 'purple', linewidth=1.5, alpha=0.8)
        ax6.scatter(vx_vel[0], vy_vel[0], color='green', s=40, label='Start')
        ax6.scatter(vx_vel[-1], vy_vel[-1], color='red', s=40, label='End')
        ax6.set_xlabel('VX Velocity')
        ax6.set_ylabel('VY Velocity')
        ax6.set_title('Phase Portrait (Velocities)')
        ax6.grid(True)
        ax6.legend()
    
    # Magnitude over time
    ax7 = fig.add_subplot(3, 3, 7)
    magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
    ax7.plot(time_axis, magnitude, 'orange', linewidth=2)
    
    if r_peak_location is not None:
        r_time = r_peak_location / beat_record.fs
        ax7.axvline(x=r_time, color='black', linestyle='--', linewidth=2, label='R-peak')
    
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('VCG Magnitude (mV)')
    ax7.set_title('VCG Vector Magnitude')
    ax7.grid(True)
    ax7.legend()
    
    # 3D trajectory with time coloring (detailed)
    ax8 = fig.add_subplot(3, 3, 8, projection='3d')
    # Use a more detailed color map
    colors_detailed = plt.cm.viridis(np.linspace(0, 1, len(vx)))
    
    for i in range(len(vx)-1):
        ax8.plot([vx[i], vx[i+1]], [vy[i], vy[i+1]], [vz[i], vz[i+1]], 
                color=colors_detailed[i], linewidth=2, alpha=0.8)
    
    ax8.scatter(vx[0], vy[0], vz[0], color='green', s=80, label='Start')
    ax8.scatter(vx[-1], vy[-1], vz[-1], color='red', s=80, label='End')
    ax8.set_title('3D Trajectory (Time Colored)')
    ax8.set_xlabel('VX')
    ax8.set_ylabel('VY')
    ax8.set_zlabel('VZ')
    ax8.legend()
    
    # Statistics
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.text(0.05, 0.95, 'Single Beat Statistics:', transform=ax9.transAxes, fontweight='bold', fontsize=12)
    ax9.text(0.05, 0.85, f'Duration: {len(vx)/beat_record.fs:.3f} seconds', transform=ax9.transAxes)
    ax9.text(0.05, 0.75, f'Samples: {len(vx)}', transform=ax9.transAxes)
    ax9.text(0.05, 0.65, f'Max VX: {vx.max():.3f} mV', transform=ax9.transAxes)
    ax9.text(0.05, 0.55, f'Max VY: {vy.max():.3f} mV', transform=ax9.transAxes)
    ax9.text(0.05, 0.45, f'Max VZ: {vz.max():.3f} mV', transform=ax9.transAxes)
    ax9.text(0.05, 0.35, f'Max Magnitude: {magnitude.max():.3f} mV', transform=ax9.transAxes)
    ax9.text(0.05, 0.25, f'VCG Loop Area (XY): {calculate_loop_area(vx, vy):.3f}', transform=ax9.transAxes)
    
    if r_peak_location is not None:
        ax9.text(0.05, 0.15, f'R-peak at: {r_peak_location/beat_record.fs:.3f}s', transform=ax9.transAxes)
    
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    
    plt.tight_layout()
    plt.show()

def calculate_loop_area(x, y):
    """Calculate the area enclosed by a 2D loop using the shoelace formula"""
    if len(x) < 3:
        return 0
    return 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))

def analyze_single_heartbeat():
    """Main function to analyze a single heartbeat"""
    
    # Load ECG and create VCG
    record_name = "/Users/alinawaf/Desktop/Research/ECG-VECG/MIMIC_Dataset/s40689238/40689238"
    record = wfdb.rdrecord(record_name)
    
    # Fix lead names and create VCG
    record_fixed = create_record_copy(record)
    record_fixed.sig_name = [name.lower() for name in record.sig_name]
    
    try:
        vcg_record = rec.kors_vcg(record_fixed)
        if vcg_record is None:
            print("Using manual VCG creation...")
            vcg_record = create_manual_vcg(record)
    except:
        print("Using manual VCG creation...")
        vcg_record = create_manual_vcg(record)
    
    # Extract different single heartbeats
    beat_numbers = [0, 1, 2, 3]  # First 4 beats
    
    for beat_num in beat_numbers:
        print(f"\n=== ANALYZING HEARTBEAT {beat_num} ===")
        single_beat, r_peak_loc = extract_single_heartbeat(vcg_record, beat_num)
        
        if single_beat is not None:
            plot_single_beat_vcg(single_beat, r_peak_loc)
        else:
            print(f"Could not extract beat {beat_num}")
            break

def create_manual_vcg(original_record):
    """Manual Kors VCG implementation"""
    lead_names = [name.lower() for name in original_record.sig_name]
    lead_map = {name: i for i, name in enumerate(lead_names)}
    
    signals = original_record.p_signal
    I = signals[:, lead_map['i']]
    II = signals[:, lead_map['ii']]
    V1 = signals[:, lead_map['v1']]
    V2 = signals[:, lead_map['v2']]
    V3 = signals[:, lead_map['v3']]
    V4 = signals[:, lead_map['v4']]
    V5 = signals[:, lead_map['v5']]
    V6 = signals[:, lead_map['v6']]
    
    VX = (0.38 * I + 0.07 * II - 0.13 * V1 + 0.05 * V2 - 0.01 * V3 + 
          0.14 * V4 + 0.06 * V5 + 0.54 * V6)
    VY = (0.11 * I - 0.23 * II + 0.06 * V1 - 0.02 * V2 + 0.05 * V3 + 
          0.06 * V4 - 0.17 * V5 + 0.13 * V6)
    VZ = (-0.08 * I + 0.06 * II - 0.36 * V1 + 0.14 * V2 - 0.27 * V3 + 
          0.21 * V4 - 0.10 * V5 + 0.06 * V6)
    
    vcg_signals = np.column_stack([VX, VY, VZ])
    
    return wfdb.Record(
        record_name='manual_vcg',
        n_sig=3,
        fs=original_record.fs,
        sig_len=len(VX),
        p_signal=vcg_signals,
        sig_name=['vx', 'vy', 'vz'],
        units=['mV', 'mV', 'mV'],
        comments=['Manual Kors VCG']
    )

if __name__ == "__main__":
    analyze_single_heartbeat()