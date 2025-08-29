import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import wfdb
import sys
import os

sys.path.append('/Users/alinawaf/Desktop/Research/ECG-VECG/CardioVectorLib')
# Now import as if cardiovector is a top-level module
from cardiovector import plotting, preprocessing, reconstruction as rec

def load_ecg_record():
    record_name = "/Users/alinawaf/Desktop/Research/ECG-VECG/MIMIC_Dataset/s40689238/40689238"
    record = wfdb.rdrecord(record_name)
    
    # Check what channels are available
    print("Available channels:", record.sig_name)
    print("Number of channels:", len(record.sig_name))
    print("Signal shape:", record.p_signal.shape)
    
    # Convert channel names to lowercase for compatibility with CardioVectorLib
    record.sig_name = [name.lower() for name in record.sig_name]
    print("Converted channels:", record.sig_name)
    
    # try:
        # Create VCG using Kors transformation
    vsgRecord = rec.kors_vcg(record)
    print("VCG created using Kors transformation.")
    plotting.plotvcg(vsgRecord, signals=['vx', 'vy', 'vz'],plot='all')
    plt.show()
    
        
        # Plot the original record and VCG
       
        
        # Plot VCG in 3D and frontal views
    # plotting.plotvcg(record, signals=['vx', 'vy', 'vz'],
    #                 plot='all')
    # plt.show()
        
    # plt.tight_layout()
    # plt.show()
        
    # except Exception as e:
    #     print(f"Error with CardioVectorLib: {e}")
    #     print("Falling back to manual VCG calculation...")
        
    #     # Fallback: Use your own VCG calculation from Plot.py
    #     from Plot import ecg_to_vcg_kors, plot_vcg_plotly, plot_vcg_matplotlib
        
    #     # Convert to DataFrame with proper column names
    #     ecg_df = pd.DataFrame(record.p_signal, columns=record.sig_name)
        
    #     # Make sure we have the required 12 leads
    #     required_leads = ['i', 'ii', 'iii', 'avr', 'avf', 'avl', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
        
    #     if all(lead in ecg_df.columns for lead in required_leads):
    #         # Reorder columns to match expected order
    #         ecg_df = ecg_df[required_leads]
            
    #         # Create VCG using your function
    #         vcg_coords = ecg_to_vcg_kors(ecg_df)
            
    #         # Plot with multiple methods
    #         plot_vcg_plotly(vcg_coords, "MIMIC Dataset VCG - Interactive")
    #         plot_vcg_matplotlib(vcg_coords, "MIMIC Dataset VCG - Static")
    #     else:
    #         missing_leads = [lead for lead in required_leads if lead not in ecg_df.columns]
    #         print(f"Missing required leads: {missing_leads}")
    #         print(f"Available leads: {list(ecg_df.columns)}")

if __name__ == "__main__":
    load_ecg_record()