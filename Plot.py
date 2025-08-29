import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import wfdb

from CardioVectorLib.cardiovector import plotting
from CardioVectorLib.cardiovector import preprocessing
from CardioVectorLib.cardiovector import preprocessing as prep
from CardioVectorLib.cardiovector import reconstruction as rec
def ecg_to_vcg_kors(ecg_data):
    """
    Convert 12-lead ECG to VCG using Kors transformation matrix.
    
    Parameters:
    ecg_data: DataFrame or array with columns [I,II,III,aVR,aVF,aVL,V1,V2,V3,V4,V5,V6]
    
    Returns:
    Dictionary with X, Y, Z coordinates
    """
    # Extract leads (assuming column order: I,II,III,aVR,aVF,aVL,V1,V2,V3,V4,V5,V6)
    if isinstance(ecg_data, pd.DataFrame):
        I = ecg_data.iloc[:, 0].values
        II = ecg_data.iloc[:, 1].values
        V1 = ecg_data.iloc[:, 6].values
        V2 = ecg_data.iloc[:, 7].values
        V3 = ecg_data.iloc[:, 8].values
        V4 = ecg_data.iloc[:, 9].values
        V5 = ecg_data.iloc[:, 10].values
        V6 = ecg_data.iloc[:, 11].values
    else:
        # Assume numpy array
        I = ecg_data[:, 0]
        II = ecg_data[:, 1]
        V1 = ecg_data[:, 6]
        V2 = ecg_data[:, 7]
        V3 = ecg_data[:, 8]
        V4 = ecg_data[:, 9]
        V5 = ecg_data[:, 10]
        V6 = ecg_data[:, 11]
    
    # Kors transformation coefficients
    # X-axis (left-right)
    X = (0.38 * I + 0.07 * II - 0.13 * V1 + 0.05 * V2 - 0.01 * V3 + 
         0.14 * V4 + 0.06 * V5 + 0.54 * V6)
    
    # Y-axis (superior-inferior) 
    Y = (0.11 * I - 0.23 * II + 0.06 * V1 - 0.02 * V2 + 0.05 * V3 + 
         0.06 * V4 - 0.17 * V5 + 0.13 * V6)
    
    # Z-axis (anterior-posterior)
    Z = (-0.08 * I + 0.06 * II - 0.36 * V1 + 0.14 * V2 - 0.27 * V3 + 
         0.21 * V4 - 0.10 * V5 + 0.06 * V6)
    
    return {'X': X, 'Y': Y, 'Z': Z}

def plot_vcg_matplotlib(vcg_coords, title="3D Vectorcardiogram"):
    """
    Plot VCG using matplotlib (basic 3D plot)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y, Z = vcg_coords['X'], vcg_coords['Y'], vcg_coords['Z']
    
    # Create color gradient based on time
    colors = np.linspace(0, 1, len(X))
    
    # Plot the VCG loop
    scatter = ax.scatter(X, Y, Z, c=colors, cmap='plasma', s=20, alpha=0.8)
    
    # Connect points with lines
    ax.plot(X, Y, Z, 'k-', alpha=0.3, linewidth=0.5)
    
    # Mark start and end points
    ax.scatter(X[0], Y[0], Z[0], color='green', s=100, label='Start')
    ax.scatter(X[-1], Y[-1], Z[-1], color='red', s=100, label='End')
    
    # Labels and formatting
    ax.set_xlabel('X (Left-Right)', fontsize=12)
    ax.set_ylabel('Y (Superior-Inferior)', fontsize=12)
    ax.set_zlabel('Z (Anterior-Posterior)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Time progression')
    
    # Legend
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_vcg_plotly(vcg_coords, title="Interactive 3D Vectorcardiogram"):
    """
    Plot VCG using Plotly (interactive 3D plot that you can rotate)
    """
    X, Y, Z = vcg_coords['X'], vcg_coords['Y'], vcg_coords['Z']
    
    # Create time-based color scale
    time_points = np.arange(len(X))
    
    # Main VCG loop
    fig = go.Figure()
    
    # Add the VCG path as a line
    fig.add_trace(go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='lines+markers',
        line=dict(
            color=time_points,
            colorscale='plasma',
            width=4
        ),
        marker=dict(
            size=3,
            color=time_points,
            colorscale='plasma',
            opacity=0.8
        ),
        name='VCG Loop',
        hovertemplate='<b>Time: %{text}</b><br>' +
                      'X: %{x:.3f}<br>' +
                      'Y: %{y:.3f}<br>' +
                      'Z: %{z:.3f}<extra></extra>',
        text=[f'{i}ms' for i in range(len(X))]
    ))
    
    # Mark start point
    fig.add_trace(go.Scatter3d(
        x=[X[0]], y=[Y[0]], z=[Z[0]],
        mode='markers',
        marker=dict(size=10, color='green'),
        name='Start'
    ))
    
    # Mark end point
    fig.add_trace(go.Scatter3d(
        x=[X[-1]], y=[Y[-1]], z=[Z[-1]],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='End'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (Left-Right)',
            yaxis_title='Y (Superior-Inferior)',
            zaxis_title='Z (Anterior-Posterior)',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600
    )
    
    fig.show()

def load_ecg_from_csv(file_path):
    """
    Load ECG data from CSV file.
    Expects 12 columns: I,II,III,aVR,aVF,aVL,V1,V2,V3,V4,V5,V6
    """
    try:
        # Since your CSV has headers, read them
        df = pd.read_csv(file_path)  # This will use the first row as headers
        print(f"CSV shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:\n{df.head()}")
        
        # Convert all columns to numeric, replacing any non-numeric values with NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        
        if df.shape[1] != 12:
            print(f"Warning: Expected 12 columns, got {df.shape[1]}")
        
        # Check for any NaN values that might indicate conversion issues
        if df.isnull().any().any():
            print("Warning: Some data could not be converted to numeric values")
            print(f"NaN count per column: {df.isnull().sum()}")
            # Drop rows with any NaN values
            df = df.dropna()
            print(f"After removing NaN rows, shape: {df.shape}")
            
        if len(df) == 0:
            print("Error: No valid data remaining after cleaning")
            return None
            
        print(f"Successfully loaded ECG data with shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
def create_sample_data():
    """Create extended sample data for better VCG visualization"""
    sample_base = np.array([
        [-0.155,0.01,0.15,0.07,0.075,-0.14,0.04,0.04,0.01,0.04,0.08,0.02],
        [-0.17,0.01,0.165,0.075,0.085,-0.16,0.055,0.055,0.02,0.03,0.08,0.02],
        [-0.17,0.0,0.155,0.08,0.075,-0.155,0.06,0.055,0.03,0.025,0.08,0.01],
        [-0.17,0.0,0.155,0.08,0.075,-0.155,0.055,0.055,0.03,0.02,0.08,0.01]
    ])
    
    # Create a more realistic ECG cycle by interpolating and adding variations
    extended_data = []
    for cycle in range(5):
        for i in range(len(sample_base)):
            # Add some realistic variation to simulate actual ECG
            variation = np.random.normal(0, 0.005, 12)  # Small noise
            cycle_variation = np.sin(cycle * 0.3) * 0.01  # Breathing variation
            
            row = sample_base[i] + variation + cycle_variation
            extended_data.append(row)
    
    return np.array(extended_data)

# === ADVANCED FEATURES ===

def plot_vcg_planes(vcg_coords):
    """Plot VCG projections on standard medical planes"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    X, Y, Z = vcg_coords['X'], vcg_coords['Y'], vcg_coords['Z']
    time_colors = np.linspace(0, 1, len(X))
    
    # Frontal plane (X-Y)
    axes[0,0].scatter(X, Y, c=time_colors, cmap='plasma', s=20)
    axes[0,0].plot(X, Y, 'k-', alpha=0.3, linewidth=0.5)
    axes[0,0].set_xlabel('X (Left-Right)')
    axes[0,0].set_ylabel('Y (Superior-Inferior)')
    axes[0,0].set_title('Frontal Plane')
    axes[0,0].grid(True, alpha=0.3)
    
    # Horizontal plane (X-Z)
    axes[0,1].scatter(X, Z, c=time_colors, cmap='plasma', s=20)
    axes[0,1].plot(X, Z, 'k-', alpha=0.3, linewidth=0.5)
    axes[0,1].set_xlabel('X (Left-Right)')
    axes[0,1].set_ylabel('Z (Anterior-Posterior)')
    axes[0,1].set_title('Horizontal Plane')
    axes[0,1].grid(True, alpha=0.3)
    
    # Sagittal plane (Y-Z)
    axes[1,0].scatter(Y, Z, c=time_colors, cmap='plasma', s=20)
    axes[1,0].plot(Y, Z, 'k-', alpha=0.3, linewidth=0.5)
    axes[1,0].set_xlabel('Y (Superior-Inferior)')
    axes[1,0].set_ylabel('Z (Anterior-Posterior)')
    axes[1,0].set_title('Sagittal Plane')
    axes[1,0].grid(True, alpha=0.3)
    
    # 3D view
    ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
    scatter = ax_3d.scatter(X, Y, Z, c=time_colors, cmap='plasma', s=20)
    ax_3d.plot(X, Y, Z, 'k-', alpha=0.3, linewidth=0.5)
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('3D VCG')
    
    plt.tight_layout()
    plt.show()

def animate_vcg_plotly(vcg_coords, interval_ms=50):
    """Create animated VCG that traces the loop over time"""
    X, Y, Z = vcg_coords['X'], vcg_coords['Y'], vcg_coords['Z']
    
    # Create frames for animation
    frames = []
    for i in range(1, len(X)+1):
        frame_data = go.Scatter3d(
            x=X[:i], y=Y[:i], z=Z[:i],
            mode='lines+markers',
            line=dict(color='red', width=6),
            marker=dict(size=4, color='red'),
            name=f'VCG Loop (t={i})'
        )
        frames.append(go.Frame(data=[frame_data], name=str(i)))
    
    # Initial plot
    fig = go.Figure(
        data=[go.Scatter3d(x=[X[0]], y=[Y[0]], z=[Z[0]], 
                          mode='markers', marker=dict(size=6, color='red'))],
        frames=frames
    )
    
    # Add animation controls
    fig.update_layout(
        title="Animated 3D Vectorcardiogram",
        scene=dict(
            xaxis_title='X (Left-Right)',
            yaxis_title='Y (Superior-Inferior)', 
            zaxis_title='Z (Anterior-Posterior)',
            aspectmode='cube'
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {'label': 'Play', 'method': 'animate',
                 'args': [None, {'frame': {'duration': interval_ms, 'redraw': True},
                               'fromcurrent': True}]},
                {'label': 'Pause', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                 'mode': 'immediate', 'transition': {'duration': 0}}]}
            ]
        }]
    )
    
    fig.show()

# === QUICK START FUNCTIONS ===

def quick_vcg_from_csv(csv_file_path):
    """
    One-line function to create VCG from CSV file
    """
    # Load data
    ecg_data = pd.read_csv(csv_file_path, header=None)
    
    # Convert to VCG
    vcg_coords = ecg_to_vcg_kors(ecg_data)
    
    # Plot interactive 3D
    plot_vcg_plotly(vcg_coords, f"VCG from {csv_file_path}")
    
    return vcg_coords

def quick_vcg_from_array(ecg_array):
    """
    One-line function to create VCG from numpy array
    """
    vcg_coords = ecg_to_vcg_kors(ecg_array)
    plot_vcg_plotly(vcg_coords, "VCG from Array Data")
    return vcg_coords

# === USAGE EXAMPLES ===

if __name__ == "__main__":
    # Enable debugging and multiple plots
    print("=== Loading ECG data from CSV ===")
    csv_path = "40689238.csv"
    ecg_df = load_ecg_from_csv(csv_path)
    
    record_name = "/Users/alinawaf/Desktop/Research/ECG-VECG/MIMIC_Dataset/s40689238/40689238"
    record = wfdb.rdrecord(record_name)

    if ecg_df is not None:
        print("\n=== Converting to VCG and plotting ===")

        vcg_coords = ecg_to_vcg_kors(ecg_df)
        
        # Plot with Plotly (interactive)
        print("Creating interactive 3D plot...")
        plot_vcg_plotly(vcg_coords, f"VCG from {csv_path}")
        
        # Plot with matplotlib (static)
        print("Creating static 3D plot...")
        plot_vcg_matplotlib(vcg_coords, f"VCG from {csv_path}")
        
        # Plot plane projections
        print("Creating plane projections...")
        plot_vcg_planes(vcg_coords)
        
    else:
        print("Failed to load CSV data, trying with sample data...")
        
        # Fallback to sample data
        print("\n=== Using sample data ===")
        sample_ecg = create_sample_data()
        vcg_coords = ecg_to_vcg_kors(sample_ecg)
        plot_vcg_plotly(vcg_coords, "Sample ECG - 3D VCG")