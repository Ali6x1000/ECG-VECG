import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import sys
import os

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

def fix_kors_vcg():
    record_name = "/Users/alinawaf/Desktop/Research/ECG-VECG/MIMIC_Dataset/s40689238/40689238"
    record = wfdb.rdrecord(record_name)
    
    print("=== ORIGINAL RECORD ===")
    print(f"Channels: {record.sig_name}")
    print(f"Shape: {record.p_signal.shape}")
    
    # The error showed CardioVectorLib expects lowercase names
    # Let's create a record with lowercase names
    print("\n=== FIXING LEAD NAMES FOR CARDIOVECTORLIB ===")
    
    record_fixed = create_record_copy(record)
    record_fixed.sig_name = [name.lower() for name in record.sig_name]
    print(f"Fixed lead names: {record_fixed.sig_name}")
    
    # Now try CardioVectorLib kors_vcg
    print("\n=== TESTING CARDIOVECTORLIB KORS_VCG ===")
    try:
        vcg_record = rec.kors_vcg(record_fixed)
        if vcg_record is not None:
            print("SUCCESS! CardioVectorLib kors_vcg worked!")
            print(f"VCG channels: {vcg_record.sig_name}")
            print(f"VCG shape: {vcg_record.p_signal.shape}")
            
            # Check VCG signal quality
            vx, vy, vz = vcg_record.p_signal[:, 0], vcg_record.p_signal[:, 1], vcg_record.p_signal[:, 2]
            print(f"VX range: {vx.max() - vx.min():.4f}")
            print(f"VY range: {vy.max() - vy.min():.4f}")
            print(f"VZ range: {vz.max() - vz.min():.4f}")
            
            return vcg_record
        else:
            print("CardioVectorLib returned None")
    except Exception as e:
        print(f"CardioVectorLib failed: {e}")
    
    # Fallback to manual implementation
    print("\n=== USING MANUAL KORS TRANSFORMATION ===")
    return create_manual_vcg(record)

def create_manual_vcg(original_record):
    """Manual Kors VCG implementation"""
    
    # Create proper lead name mapping
    lead_names = [name.lower() for name in original_record.sig_name]
    lead_map = {name: i for i, name in enumerate(lead_names)}
    
    print(f"Lead mapping: {lead_map}")
    
    # Extract signals using the mapping
    signals = original_record.p_signal
    I = signals[:, lead_map['i']]
    II = signals[:, lead_map['ii']]
    V1 = signals[:, lead_map['v1']]
    V2 = signals[:, lead_map['v2']]
    V3 = signals[:, lead_map['v3']]
    V4 = signals[:, lead_map['v4']]
    V5 = signals[:, lead_map['v5']]
    V6 = signals[:, lead_map['v6']]
    
    # Kors transformation
    VX = (0.38 * I + 0.07 * II - 0.13 * V1 + 0.05 * V2 - 0.01 * V3 + 
          0.14 * V4 + 0.06 * V5 + 0.54 * V6)
    
    VY = (0.11 * I - 0.23 * II + 0.06 * V1 - 0.02 * V2 + 0.05 * V3 + 
          0.06 * V4 - 0.17 * V5 + 0.13 * V6)
    
    VZ = (-0.08 * I + 0.06 * II - 0.36 * V1 + 0.14 * V2 - 0.27 * V3 + 
          0.21 * V4 - 0.10 * V5 + 0.06 * V6)
    
    print(f"VCG ranges - VX: {VX.max()-VX.min():.4f}, VY: {VY.max()-VY.min():.4f}, VZ: {VZ.max()-VZ.min():.4f}")
    
    # Create VCG record
    vcg_signals = np.column_stack([VX, VY, VZ])
    
    vcg_record = wfdb.Record(
        record_name='manual_vcg',
        n_sig=3,
        fs=original_record.fs,
        sig_len=len(VX),
        p_signal=vcg_signals,
        sig_name=['vx', 'vy', 'vz'],
        units=['mV', 'mV', 'mV'],
        comments=['Manual Kors VCG']
    )
    
    return vcg_record

def plot_working_vcg(vcg_record):
    """Plot the VCG using both manual and CardioVectorLib methods"""
    
    if vcg_record is None:
        print("No VCG record available")
        return
    
    print(f"\n=== PLOTTING VCG ===")
    
    vx, vy, vz = vcg_record.p_signal[:, 0], vcg_record.p_signal[:, 1], vcg_record.p_signal[:, 2]
    
    # Manual comprehensive plot
    fig = plt.figure(figsize=(16, 12))
    
    # Time domain
    ax1 = fig.add_subplot(3, 3, 1)
    time = np.arange(len(vx)) / vcg_record.fs
    ax1.plot(time, vx, 'r-', label='VX', linewidth=1)
    ax1.plot(time, vy, 'g-', label='VY', linewidth=1)
    ax1.plot(time, vz, 'b-', label='VZ', linewidth=1)
    ax1.set_title('VCG Time Series')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (mV)')
    ax1.legend()
    ax1.grid(True)
    
    # 3D plot - full dataset
    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    colors_full = np.linspace(0, 1, len(vx))
    ax2.scatter(vx, vy, vz, c=colors_full, cmap='plasma', s=1, alpha=0.6)
    ax2.plot(vx, vy, vz, 'k-', alpha=0.2, linewidth=0.3)
    ax2.set_title('3D VCG - Full Data')
    ax2.set_xlabel('VX')
    ax2.set_ylabel('VY')
    ax2.set_zlabel('VZ')
    
    # 3D plot - first 2 seconds for detail
    ax3 = fig.add_subplot(3, 3, 3, projection='3d')
    n_detail = min(1000, len(vx))  # First 2 seconds at 500Hz
    colors_detail = np.linspace(0, 1, n_detail)
    ax3.scatter(vx[:n_detail], vy[:n_detail], vz[:n_detail], c=colors_detail, cmap='plasma', s=10, alpha=0.8)
    ax3.plot(vx[:n_detail], vy[:n_detail], vz[:n_detail], 'k-', alpha=0.4, linewidth=0.8)
    ax3.scatter(vx[0], vy[0], vz[0], color='green', s=50, label='Start')
    ax3.scatter(vx[n_detail-1], vy[n_detail-1], vz[n_detail-1], color='red', s=50, label='End')
    ax3.set_title('3D VCG - Detail (2s)')
    ax3.set_xlabel('VX')
    ax3.set_ylabel('VY')
    ax3.set_zlabel('VZ')
    ax3.legend()
    
    # Frontal plane (XY)
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(vx, vy, 'b-', linewidth=0.8, alpha=0.7)
    ax4.scatter(vx[0], vy[0], color='green', s=30, label='Start')
    ax4.scatter(vx[-1], vy[-1], color='red', s=30, label='End')
    ax4.set_title('Frontal Plane (XY)')
    ax4.set_xlabel('VX (Left-Right)')
    ax4.set_ylabel('VY (Superior-Inferior)')
    ax4.grid(True)
    ax4.legend()
    ax4.axis('equal')
    
    # Horizontal plane (XZ)
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(vx, vz, 'g-', linewidth=0.8, alpha=0.7)
    ax5.scatter(vx[0], vz[0], color='green', s=30, label='Start')
    ax5.scatter(vx[-1], vz[-1], color='red', s=30, label='End')
    ax5.set_title('Horizontal Plane (XZ)')
    ax5.set_xlabel('VX (Left-Right)')
    ax5.set_ylabel('VZ (Anterior-Posterior)')
    ax5.grid(True)
    ax5.legend()
    ax5.axis('equal')
    
    # Sagittal plane (YZ)
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(vy, vz, 'r-', linewidth=0.8, alpha=0.7)
    ax6.scatter(vy[0], vz[0], color='green', s=30, label='Start')
    ax6.scatter(vy[-1], vz[-1], color='red', s=30, label='End')
    ax6.set_title('Sagittal Plane (YZ)')
    ax6.set_xlabel('VY (Superior-Inferior)')
    ax6.set_ylabel('VZ (Anterior-Posterior)')
    ax6.grid(True)
    ax6.legend()
    ax6.axis('equal')
    
    # Single heartbeat detail
    ax7 = fig.add_subplot(3, 3, 7)
    beat_samples = int(1.2 * vcg_record.fs)  # 1.2 seconds for one beat
    beat_time = np.arange(beat_samples) / vcg_record.fs
    ax7.plot(beat_time, vx[:beat_samples], 'r-', label='VX', linewidth=2)
    ax7.plot(beat_time, vy[:beat_samples], 'g-', label='VY', linewidth=2)
    ax7.plot(beat_time, vz[:beat_samples], 'b-', label='VZ', linewidth=2)
    ax7.set_title('Single Heartbeat Detail')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Amplitude (mV)')
    ax7.legend()
    ax7.grid(True)
    
    # Single beat 3D
    ax8 = fig.add_subplot(3, 3, 8, projection='3d')
    beat_colors = np.linspace(0, 1, beat_samples)
    ax8.scatter(vx[:beat_samples], vy[:beat_samples], vz[:beat_samples], 
               c=beat_colors, cmap='plasma', s=15, alpha=0.8)
    ax8.plot(vx[:beat_samples], vy[:beat_samples], vz[:beat_samples], 'k-', linewidth=1.5)
    ax8.set_title('Single Beat 3D')
    ax8.set_xlabel('VX')
    ax8.set_ylabel('VY')
    ax8.set_zlabel('VZ')
    
    # Statistics
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.text(0.1, 0.9, f'VCG Statistics:', transform=ax9.transAxes, fontweight='bold')
    ax9.text(0.1, 0.8, f'Duration: {len(vx)/vcg_record.fs:.1f} seconds', transform=ax9.transAxes)
    ax9.text(0.1, 0.7, f'Samples: {len(vx)}', transform=ax9.transAxes)
    ax9.text(0.1, 0.6, f'Sampling rate: {vcg_record.fs} Hz', transform=ax9.transAxes)
    ax9.text(0.1, 0.5, f'VX range: {vx.max()-vx.min():.4f} mV', transform=ax9.transAxes)
    ax9.text(0.1, 0.4, f'VY range: {vy.max()-vy.min():.4f} mV', transform=ax9.transAxes)
    ax9.text(0.1, 0.3, f'VZ range: {vz.max()-vz.min():.4f} mV', transform=ax9.transAxes)
    ax9.text(0.1, 0.2, f'Heart beats: ~{len(vx)/(1.0*vcg_record.fs):.0f}', transform=ax9.transAxes)
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Now try CardioVectorLib plotting
    print("\n=== CARDIOVECTORLIB PLOTTING TEST ===")
    try:
        plotting.plotvcg(vcg_record, 
                        signals=['vx', 'vy', 'vz'], 
                        plot='3d',
                        figsize=8)
        plt.title('CardioVectorLib 3D VCG')
        plt.show()
        
        plotting.plotvcg(vcg_record, 
                        signals=['vx', 'vy', 'vz'], 
                        plot='all',
                        figsize=6)
        plt.suptitle('CardioVectorLib - All VCG Views')
        plt.show()
        
        print("SUCCESS: CardioVectorLib plotting worked!")
        
    except Exception as e:
        print(f"CardioVectorLib plotting failed: {e}")
        print("But manual plotting above shows proper VCG loops!")

if __name__ == "__main__":
    vcg_record = fix_kors_vcg()
    plot_working_vcg(vcg_record)