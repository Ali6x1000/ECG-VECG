import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import sys
import os

sys.path.append('/Users/alinawaf/Desktop/Research/ECG-VECG/CardioVectorLib')
# Import CardioVectorLib modules
from cardiovector import plotting, preprocessing, reconstruction as rec

def load_ecg_record():
    record_name = "/Users/alinawaf/Desktop/Research/ECG-VECG/MIMIC_Dataset/s40689238/40689238"
    record = wfdb.rdrecord(record_name)
    
    # Check what channels are available
    print("Available channels:", record.sig_name)
    print("Number of channels:", len(record.sig_name))
    print("Signal shape:", record.p_signal.shape)
    
    # Convert channel names to lowercase for CardioVectorLib compatibility
    record.sig_name = [name.lower() for name in record.sig_name]
    print("Converted channels:", record.sig_name)
    
    # Create VCG using Kors transformation from CardioVectorLib
    vcg_record = rec.kors_vcg(record)
    print("VCG created using Kors transformation.")
    print("VCG record channels:", vcg_record.sig_name)
    
    # Optional: Apply preprocessing (remove baseline wandering, slice)
    filtered_record = preprocessing.remove_baseline_wandering(vcg_record)
    sliced_record = preprocessing.recslice(filtered_record, sampfrom=0, sampto=3000)
    
    # Plot VCG using CardioVectorLib - 3D and frontal view
    print("Plotting VCG in 3D and frontal view...")
    plotting.plotvcg(sliced_record, 
                    signals=['vx', 'vy', 'vz'],
                    plot=['3d', 'frontal'])
    plt.tight_layout()
    plt.show()
    
    # Plot all VCG views (3D + all 2D projections)
    print("Plotting all VCG views...")
    plotting.plotvcg(sliced_record, 
                    signals=['vx', 'vy', 'vz'],
                    plot='all')
    plt.show()
    
    # Optional: Plot original ECG record for comparison
    print("Plotting original ECG record...")
    plotting.plotrecs([record], 
                     signals=[['i', 'ii', 'v1', 'v2', 'v3', 'v4']],
                     labels=['Original 12-lead ECG'],
                     fig_kw={'figsize': (15, 8)})
    plt.show()
    
    # Optional: Compare original ECG with VCG
    print("Plotting ECG vs VCG comparison...")
    plotting.plotrecs([record, sliced_record],
                     signals=[['i', 'ii', 'v1'], ['vx', 'vy', 'vz']],
                     labels=['Original ECG', 'Kors VCG'],
                     fig_kw={'figsize': (12, 7)})
    plt.show()

if __name__ == "__main__":
    load_ecg_record()