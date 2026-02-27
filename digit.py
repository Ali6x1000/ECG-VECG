#!/usr/bin/env python3
"""
ECG PNG to WFDB Converter
Digitizes ECG traces from PNG images and converts to WFDB format
"""

import cv2
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import argparse
import os
from typing import List, Tuple, Dict
import json

class ECGDigitizer:
    def __init__(self, sampling_rate: int = 500):
        """
        Initialize ECG digitizer
        
        Args:
            sampling_rate: Target sampling rate for output signals (Hz)
        """
        self.sampling_rate = sampling_rate
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess the ECG image
        
        Args:
            image_path: Path to the ECG PNG file
            
        Returns:
            Preprocessed grayscale image
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def detect_grid_and_calibration(self, image: np.ndarray) -> Dict:
        """
        Detect ECG grid and calibration information
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Dictionary containing grid and calibration info
        """
        height, width = image.shape
        
        # Detect horizontal and vertical lines (grid)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width//20, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height//20))
        
        # Find horizontal lines
        horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=2)
        
        # Find vertical lines  
        vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=2)
        
        # Standard ECG paper: 25mm/s, 10mm/mV
        # Estimate pixels per mm based on grid detection
        h_lines_y = np.where(np.mean(horizontal_lines, axis=1) > 50)[0]
        v_lines_x = np.where(np.mean(vertical_lines, axis=0) > 50)[0]
        
        if len(h_lines_y) > 1 and len(v_lines_x) > 1:
            pixels_per_mm_y = np.median(np.diff(h_lines_y))
            pixels_per_mm_x = np.median(np.diff(v_lines_x))
        else:
            # Default fallback values
            pixels_per_mm_y = height / 100  # Assume 100mm height
            pixels_per_mm_x = width / 250   # Assume 250mm width (10s at 25mm/s)
            
        return {
            'pixels_per_mm_x': pixels_per_mm_x,
            'pixels_per_mm_y': pixels_per_mm_y,
            'horizontal_lines': h_lines_y,
            'vertical_lines': v_lines_x,
            'image_width': width,
            'image_height': height
        }
    
    def segment_leads(self, image: np.ndarray, num_leads: int = 12) -> List[np.ndarray]:
        """
        Segment the image into individual lead strips
        
        Args:
            image: Preprocessed image
            num_leads: Number of leads to extract
            
        Returns:
            List of image segments for each lead
        """
        height, width = image.shape
        
        # Common ECG layouts
        if num_leads == 12:
            # Standard 12-lead layout: 4 rows of 3 leads + rhythm strip
            rows = 4
            leads_per_row = 3
        elif num_leads == 6:
            rows = 2
            leads_per_row = 3
        elif num_leads == 3:
            rows = 3
            leads_per_row = 1
        else:
            rows = num_leads
            leads_per_row = 1
            
        row_height = height // rows
        lead_segments = []
        
        for row in range(rows):
            y_start = row * row_height
            y_end = (row + 1) * row_height
            
            if row < rows - 1:  # Not the last row (rhythm strip)
                col_width = width // leads_per_row
                for col in range(leads_per_row):
                    x_start = col * col_width
                    x_end = (col + 1) * col_width
                    segment = image[y_start:y_end, x_start:x_end]
                    lead_segments.append(segment)
            else:
                # Last row is typically a rhythm strip (full width)
                segment = image[y_start:y_end, :]
                lead_segments.append(segment)
                
        return lead_segments[:num_leads]
    
    def extract_signal_from_segment(self, segment: np.ndarray, calibration: Dict) -> np.ndarray:
        """
        Extract the signal trace from a lead segment
        
        Args:
            segment: Image segment containing one lead
            calibration: Calibration information
            
        Returns:
            1D array of signal values
        """
        height, width = segment.shape
        
        # Apply threshold to isolate the signal trace
        # ECG traces are typically dark on light background
        thresh = cv2.threshold(segment, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Invert if needed (signal should be white on black)
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)
            
        # Find the signal trace for each x position
        signal_y = []
        
        for x in range(width):
            column = thresh[:, x]
            white_pixels = np.where(column == 255)[0]
            
            if len(white_pixels) > 0:
                # Use median y position of white pixels
                y_pos = np.median(white_pixels)
            else:
                # Interpolate from neighboring columns
                y_pos = height // 2  # Default to middle
                
            signal_y.append(y_pos)
        
        # Convert pixel positions to voltage values
        # Invert y-axis (top of image = positive voltage)
        signal_y = np.array(signal_y)
        signal_y = height - signal_y
        
        # Convert to mV using calibration
        # Standard: 10mm = 1mV
        pixels_per_mv = calibration['pixels_per_mm_y'] * 10
        signal_mv = (signal_y - height/2) / pixels_per_mv
        
        # Smooth the signal
        signal_mv = signal.savgol_filter(signal_mv, window_length=5, polyorder=2)
        
        return signal_mv
    
    def resample_signal(self, signal_data: np.ndarray, original_duration: float) -> np.ndarray:
        """
        Resample signal to target sampling rate
        
        Args:
            signal_data: Original signal data
            original_duration: Duration of original signal in seconds
            
        Returns:
            Resampled signal data
        """
        original_length = len(signal_data)
        target_length = int(original_duration * self.sampling_rate)
        
        # Create interpolation function
        x_original = np.linspace(0, original_duration, original_length)
        x_target = np.linspace(0, original_duration, target_length)
        
        interpolator = interp1d(x_original, signal_data, kind='cubic', 
                              bounds_error=False, fill_value='extrapolate')
        resampled = interpolator(x_target)
        
        return resampled
    
    def digitize_ecg(self, image_path: str, duration: float = 10.0, 
                    num_leads: int = 12, output_name: str = None) -> str:
        """
        Main function to digitize ECG from PNG and save as WFDB
        
        Args:
            image_path: Path to ECG PNG file
            duration: Duration of ECG recording in seconds
            num_leads: Number of leads to extract
            output_name: Output filename (without extension)
            
        Returns:
            Path to created WFDB files
        """
        if output_name is None:
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            
        print(f"Processing ECG image: {image_path}")
        
        # Preprocess image
        image = self.preprocess_image(image_path)
        print(f"Image loaded: {image.shape}")
        
        # Detect calibration
        calibration = self.detect_grid_and_calibration(image)
        print(f"Calibration detected - X: {calibration['pixels_per_mm_x']:.1f} px/mm, "
              f"Y: {calibration['pixels_per_mm_y']:.1f} px/mm")
        
        # Segment leads
        lead_segments = self.segment_leads(image, num_leads)
        print(f"Segmented into {len(lead_segments)} leads")
        
        # Extract signals from each lead
        signals = []
        lead_names_used = self.lead_names[:num_leads]
        
        for i, segment in enumerate(lead_segments):
            print(f"Extracting signal from lead {lead_names_used[i]}...")
            signal_data = self.extract_signal_from_segment(segment, calibration)
            
            # Resample to target sampling rate
            resampled_signal = self.resample_signal(signal_data, duration)
            signals.append(resampled_signal)
        
        # Convert to numpy array (leads x samples)
        signals_array = np.array(signals)
        
        # Create WFDB record
        self.save_wfdb_record(signals_array, lead_names_used, output_name, duration)
        
        print(f"WFDB files created: {output_name}.dat, {output_name}.hea")
        return output_name
    
    def save_wfdb_record(self, signals: np.ndarray, lead_names: List[str], 
                        record_name: str, duration: float):
        """
        Save signals as WFDB format files
        
        Args:
            signals: Signal data (leads x samples)
            lead_names: Names of the leads
            record_name: Output record name
            duration: Signal duration in seconds
        """
        # WFDB expects signals in (samples x leads) format
        signals_transposed = signals.T
        
        # Create signal info for each lead
        sig_names = lead_names
        units = ['mV'] * len(lead_names)
        
        # Write WFDB record
        wfdb.wrsamp(
            record_name=record_name,
            fs=self.sampling_rate,
            units=units,
            sig_name=sig_names,
            p_signal=signals_transposed,
            fmt=['16'] * len(lead_names),
            comments=[f'Digitized from PNG image', f'Duration: {duration}s']
        )
    
    def validate_output(self, record_name: str):
        """
        Validate the created WFDB files by reading them back
        
        Args:
            record_name: Name of the WFDB record to validate
        """
        try:
            record = wfdb.rdrecord(record_name)
            print(f"\nValidation successful!")
            print(f"Record: {record_name}")
            print(f"Sampling rate: {record.fs} Hz")
            print(f"Duration: {record.sig_len / record.fs:.1f} seconds")
            print(f"Leads: {record.sig_name}")
            print(f"Signal shape: {record.p_signal.shape}")
            
            # Plot first few seconds for visual verification
            self.plot_validation(record)
            
        except Exception as e:
            print(f"Validation failed: {e}")
    
    def plot_validation(self, record, duration_to_plot: float = 3.0):
        """
        Plot the first few seconds of digitized signals for validation
        
        Args:
            record: WFDB record object
            duration_to_plot: Duration to plot in seconds
        """
        samples_to_plot = int(duration_to_plot * record.fs)
        time_axis = np.arange(samples_to_plot) / record.fs
        
        fig, axes = plt.subplots(len(record.sig_name), 1, figsize=(12, 2*len(record.sig_name)))
        if len(record.sig_name) == 1:
            axes = [axes]
            
        for i, lead_name in enumerate(record.sig_name):
            axes[i].plot(time_axis, record.p_signal[:samples_to_plot, i])
            axes[i].set_title(f'Lead {lead_name}')
            axes[i].set_ylabel('Amplitude (mV)')
            axes[i].grid(True, alpha=0.3)
            
        axes[-1].set_xlabel('Time (seconds)')
        plt.tight_layout()
        plt.savefig(f'{record.record_name}_validation.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Digitize ECG PNG files to WFDB format')
    parser.add_argument('input_image', help='Path to input ECG PNG file')
    parser.add_argument('-o', '--output', help='Output filename (without extension)')
    parser.add_argument('-d', '--duration', type=float, default=10.0, 
                       help='Duration of ECG recording in seconds (default: 10.0)')
    parser.add_argument('-n', '--num_leads', type=int, default=12,
                       help='Number of leads to extract (default: 12)')
    parser.add_argument('-fs', '--sampling_rate', type=int, default=500,
                       help='Sampling rate in Hz (default: 500)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate output by plotting signals')
    
    args = parser.parse_args()
    
    # Initialize digitizer
    digitizer = ECGDigitizer(sampling_rate=args.sampling_rate)
    
    try:
        # Process the ECG image
        output_name = digitizer.digitize_ecg(
            image_path=args.input_image,
            duration=args.duration,
            num_leads=args.num_leads,
            output_name=args.output
        )
        
        # Validate if requested
        if args.validate:
            digitizer.validate_output(output_name)
            
        print(f"\nSuccessfully digitized ECG to WFDB format: {output_name}")
        
    except Exception as e:
        print(f"Error processing ECG: {e}")
        return 1
        
    return 0


# Example usage functions
def batch_process_ecgs(input_directory: str, output_directory: str = None):
    """
    Process multiple ECG PNG files in a directory
    
    Args:
        input_directory: Directory containing ECG PNG files
        output_directory: Output directory for WFDB files
    """
    if output_directory is None:
        output_directory = input_directory
        
    os.makedirs(output_directory, exist_ok=True)
    digitizer = ECGDigitizer()
    
    png_files = [f for f in os.listdir(input_directory) if f.lower().endswith('.png')]
    
    for png_file in png_files:
        input_path = os.path.join(input_directory, png_file)
        output_name = os.path.join(output_directory, os.path.splitext(png_file)[0])
        
        try:
            digitizer.digitize_ecg(input_path, output_name=output_name)
            print(f"Processed: {png_file}")
        except Exception as e:
            print(f"Failed to process {png_file}: {e}")


def create_sample_config():
    """Create a sample configuration file for advanced settings"""
    config = {
        "sampling_rate": 500,
        "duration": 10.0,
        "num_leads": 12,
        "lead_layout": {
            "rows": 4,
            "leads_per_row": 3,
            "rhythm_strip": True
        },
        "calibration": {
            "time_scale": 25,  # mm/s
            "voltage_scale": 10,  # mm/mV
            "auto_detect": True
        },
        "preprocessing": {
            "gaussian_blur": True,
            "contrast_enhancement": True,
            "noise_reduction": True
        }
    }
    
    with open('ecg_digitizer_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Sample configuration file created: ecg_digitizer_config.json")


if __name__ == "__main__":
    import sys
    
    # Check if required packages are available
    required_packages = ['cv2', 'wfdb', 'scipy', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install opencv-python wfdb scipy matplotlib")
        sys.exit(1)
    
    # Run main function
    sys.exit(main())