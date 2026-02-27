import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter1d
import sys
sys.path.append('/Users/alinawaf/Desktop/Research/ECG-VECG/CardioVectorLib')
from cardiovector import plotting, preprocessing, reconstruction as rec

def robust_qrs_detection(signal_data, fs=500, lead_name=""):
    """
    Robust QRS detection for scanned ECG with multiple fallback strategies
    """
    # Strategy 1: Adaptive threshold based on signal characteristics
    smoothed = signal.savgol_filter(signal_data, min(51, len(signal_data)//4), 3)
    
    # Calculate multiple signal metrics
    signal_std = np.std(smoothed)
    signal_mean = np.mean(smoothed)
    signal_abs_mean = np.mean(np.abs(smoothed - signal_mean))
    
    # Try multiple detection strategies
    strategies = [
        {'height': signal_std * 0.5, 'prominence': signal_std * 0.3},  # Very permissive
        {'height': signal_std * 1.0, 'prominence': signal_std * 0.5},  # Moderate
        {'height': signal_abs_mean * 1.5, 'prominence': signal_abs_mean * 0.8},  # Alternative metric
    ]
    
    best_peaks = []
    best_polarity = "positive"
    
    for strategy in strategies:
        # Try positive peaks
        pos_peaks, pos_props = find_peaks(smoothed, 
                                         height=strategy['height'],
                                         prominence=strategy['prominence'],
                                         distance=int(0.25 * fs))  # 250ms min distance
        
        # Try negative peaks
        neg_peaks, neg_props = find_peaks(-smoothed, 
                                         height=strategy['height'],
                                         prominence=strategy['prominence'],
                                         distance=int(0.25 * fs))
        
        # Evaluate which gives better results
        pos_score = len(pos_peaks) * np.mean(pos_props['peak_heights']) if len(pos_peaks) > 0 else 0
        neg_score = len(neg_peaks) * np.mean(neg_props['peak_heights']) if len(neg_peaks) > 0 else 0
        
        if pos_score > neg_score and len(pos_peaks) >= 2:
            best_peaks = pos_peaks
            best_polarity = "positive"
            break
        elif neg_score > pos_score and len(neg_peaks) >= 2:
            best_peaks = neg_peaks
            best_polarity = "negative"
            break
    
    # If no good peaks found, try derivative-based detection
    if len(best_peaks) < 2:
        diff_signal = np.abs(np.diff(smoothed))
        diff_peaks, _ = find_peaks(diff_signal, 
                                  height=np.percentile(diff_signal, 75),
                                  distance=int(0.25 * fs))
        
        if len(diff_peaks) >= 2:
            best_peaks = diff_peaks
            best_polarity = "derivative"
    
    return best_peaks, best_polarity

def cross_correlation_sync(ref_signal, target_signal, max_shift=250):
    """
    Cross-correlation based synchronization with larger search window
    """
    # Normalize signals
    ref_norm = (ref_signal - np.mean(ref_signal)) / np.std(ref_signal)
    target_norm = (target_signal - np.mean(target_signal)) / np.std(target_signal)
    
    # Try both normal and inverted correlation
    corr_normal = np.correlate(ref_norm, target_norm, mode='full')
    corr_inverted = np.correlate(ref_norm, -target_norm, mode='full')
    
    # Find best correlation
    best_normal = np.max(corr_normal)
    best_inverted = np.max(corr_inverted)
    
    if best_inverted > best_normal:
        correlation = corr_inverted
        polarity = "inverted"
    else:
        correlation = corr_normal  
        polarity = "normal"
    
    # Find shift
    correlation_center = len(correlation) // 2
    peak_idx = np.argmax(correlation)
    shift = peak_idx - correlation_center
    
    # Limit shift to reasonable range
    if abs(shift) > max_shift:
        shift = 0
        correlation_value = 0
    else:
        correlation_value = correlation[peak_idx] / len(ref_signal)
    
    return shift, correlation_value, polarity

def robust_ecg_synchronization():
    """
    Robust ECG synchronization using multiple alignment strategies
    """
    print("=== ROBUST ECG SYNCHRONIZATION FOR SCANNED DATA ===")
    
    # Load data
    Path_Name = "/Users/alinawaf/Desktop/Research/ECG-VECG/Output/1/00001_-0_0000"
    record = wfdb.rdrecord(Path_Name, physical=False)
    record.base_datetime = None
    record.sig_name = [s.lower() for s in record.sig_name]
    
    # Apply baseline filtering
    try:
        filtered_record = preprocessing.remove_baseline_wandering(record)
        signals = filtered_record.d_signal
        print("✅ Baseline filtering applied")
    except:
        signals = record.d_signal
        filtered_record = record
        print("⚠️ Using unfiltered signals")
    
    # Use common period
    common_length = 1300  # 2.6 seconds
    sync_signals = signals[:common_length, :].copy()
    lead_names = record.sig_name
    
    print(f"Using {common_length} samples ({common_length/500:.1f}s) for synchronization")
    
    # Step 1: Robust QRS detection in all leads
    print(f"\n=== ROBUST QRS DETECTION ===")
    lead_qrs_data = {}
    
    for i, lead_name in enumerate(lead_names):
        peaks, polarity = robust_qrs_detection(sync_signals[:, i], fs=500, lead_name=lead_name)
        lead_qrs_data[lead_name] = {
            'peaks': peaks,
            'polarity': polarity,
            'signal': sync_signals[:, i]
        }
        print(f"  {lead_name.upper()}: {len(peaks)} QRS detected, polarity: {polarity}")
    
    # Step 2: Choose reference lead (Lead II preferred, or lead with most QRS)
    if 'ii' in lead_qrs_data and len(lead_qrs_data['ii']['peaks']) >= 2:
        ref_lead = 'ii'
    else:
        # Find lead with most reliable QRS detection
        qrs_counts = {lead: len(data['peaks']) for lead, data in lead_qrs_data.items()}
        ref_lead = max(qrs_counts, key=qrs_counts.get) if qrs_counts else 'ii'
    
    ref_signal = lead_qrs_data[ref_lead]['signal']
    ref_peaks = lead_qrs_data[ref_lead]['peaks']
    print(f"\nUsing {ref_lead.upper()} as reference ({len(ref_peaks)} QRS complexes)")
    
    # Step 3: Synchronization using multiple methods
    print(f"\n=== MULTI-METHOD SYNCHRONIZATION ===")
    synchronized_signals = sync_signals.copy()
    sync_results = {}
    
    for i, lead_name in enumerate(lead_names):
        if lead_name == ref_lead:
            sync_results[lead_name] = {'shift': 0, 'correlation': 1.0, 'method': 'reference'}
            print(f"  {lead_name.upper()}: Reference lead")
            continue
        
        lead_signal = lead_qrs_data[lead_name]['signal']
        lead_peaks = lead_qrs_data[lead_name]['peaks']
        
        best_shift = 0
        best_correlation = 0
        best_method = "none"
        
        # Method 1: QRS peak alignment (if both leads have QRS)
        if len(ref_peaks) > 0 and len(lead_peaks) > 0:
            qrs_shift = ref_peaks[0] - lead_peaks[0]
            
            # Test this shift with cross-correlation
            if abs(qrs_shift) <= 400:  # 800ms max shift
                # Create shifted version for testing
                if qrs_shift > 0:
                    test_signal = np.concatenate([np.zeros(qrs_shift), lead_signal[:-qrs_shift]])
                elif qrs_shift < 0:
                    test_signal = np.concatenate([lead_signal[-qrs_shift:], np.zeros(-qrs_shift)])
                else:
                    test_signal = lead_signal
                
                # Calculate correlation
                correlation = np.corrcoef(ref_signal, test_signal)[0, 1]
                
                if abs(correlation) > 0.3:  # Reasonable correlation
                    best_shift = qrs_shift
                    best_correlation = correlation
                    best_method = "qrs_alignment"
        
        # Method 2: Cross-correlation sync (fallback)
        if abs(best_correlation) < 0.3:
            shift, correlation, polarity = cross_correlation_sync(ref_signal, lead_signal, max_shift=400)
            
            if abs(correlation) > abs(best_correlation):
                best_shift = shift
                best_correlation = correlation
                best_method = f"cross_correlation_{polarity}"
        
        # Apply synchronization if improvement found
        if abs(best_correlation) > 0.2 and best_shift != 0:
            # Apply the shift
            if best_shift > 0:
                synchronized_signals[best_shift:, i] = sync_signals[:-best_shift, i]
                synchronized_signals[:best_shift, i] = sync_signals[0, i]  # Pad with first value
            elif best_shift < 0:
                synchronized_signals[:best_shift, i] = sync_signals[-best_shift:, i]  
                synchronized_signals[best_shift:, i] = sync_signals[-1, i]  # Pad with last value
            
            shift_ms = best_shift * 1000 / 500
            print(f"  {lead_name.upper()}: {best_method}, shift {shift_ms:.0f}ms, correlation: {best_correlation:.3f}")
        else:
            print(f"  {lead_name.upper()}: No reliable sync found (corr: {best_correlation:.3f})")
        
        sync_results[lead_name] = {
            'shift': best_shift,
            'correlation': best_correlation,
            'method': best_method
        }
    
    # Step 4: Quality assessment
    print(f"\n=== SYNCHRONIZATION QUALITY ASSESSMENT ===")
    
    def calculate_lead_group_correlation(signals, lead_names, group_leads):
        correlations = []
        group_indices = [i for i, name in enumerate(lead_names) if name in group_leads]
        
        for i in range(len(group_indices)):
            for j in range(i+1, len(group_indices)):
                idx1, idx2 = group_indices[i], group_indices[j]
                corr = np.corrcoef(signals[:, idx1], signals[:, idx2])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))  # Use absolute correlation
        
        return np.mean(correlations) if correlations else 0.0
    
    # Lead groups
    limb_leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf']
    precordial_leads = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    
    # Before synchronization
    limb_corr_before = calculate_lead_group_correlation(sync_signals, lead_names, limb_leads)
    prec_corr_before = calculate_lead_group_correlation(sync_signals, lead_names, precordial_leads)
    overall_before = calculate_lead_group_correlation(sync_signals, lead_names, lead_names)
    
    # After synchronization  
    limb_corr_after = calculate_lead_group_correlation(synchronized_signals, lead_names, limb_leads)
    prec_corr_after = calculate_lead_group_correlation(synchronized_signals, lead_names, precordial_leads)
    overall_after = calculate_lead_group_correlation(synchronized_signals, lead_names, lead_names)
    
    print(f"Overall correlation:")
    print(f"  Before: {overall_before:.3f}")
    print(f"  After:  {overall_after:.3f}")
    print(f"  Change: {overall_after - overall_before:+.3f}")
    
    print(f"Limb leads correlation:")
    print(f"  Before: {limb_corr_before:.3f}")
    print(f"  After:  {limb_corr_after:.3f}")
    print(f"  Change: {limb_corr_after - limb_corr_before:+.3f}")
    
    print(f"Precordial leads correlation:")
    print(f"  Before: {prec_corr_before:.3f}")
    print(f"  After:  {prec_corr_after:.3f}")
    print(f"  Change: {prec_corr_after - prec_corr_before:+.3f}")
    
    # Count successful synchronizations 
    successful_syncs = sum(1 for result in sync_results.values() 
                          if abs(result['correlation']) > 0.2 and result['shift'] != 0)
    
    # Step 5: Generate VCG from synchronized data
    print(f"\n=== GENERATING VCG FROM SYNCHRONIZED DATA ===")
    
    # Create synchronized record by copying the original filtered record structure
    try:
        # Use the filtered record as template and update with synchronized data
        sync_record = filtered_record
        sync_record.d_signal = synchronized_signals.astype(np.int16)
        sync_record.sig_len = len(synchronized_signals)
        # Keep lead names in lowercase for cardiovector library compatibility
        sync_record.sig_name = [name.lower() for name in lead_names]
        
        print(f"Using original record structure as template")
        print(f"Sync record shape: {sync_record.d_signal.shape}")
        print(f"Lead names: {sync_record.sig_name}")  # Should show lowercase
        
        vcg_record = rec.kors_vcg(sync_record)
        
        if vcg_record is None:
            print("❌ VCG record is None - check KORS transform")
            return None
            
        if not hasattr(vcg_record, 'd_signal') or vcg_record.d_signal is None:
            print("❌ VCG record has no d_signal attribute")
            return None
            
        vcg_signals = vcg_record.d_signal
        print(f"VCG signals shape: {vcg_signals.shape}")
        
        if vcg_signals.shape[1] != 3:
            print(f"❌ Expected 3 VCG components, got {vcg_signals.shape[1]}")
            return None
            
        vx, vy, vz = vcg_signals.T
        
        # Plot comprehensive VCG
        fig = plt.figure(figsize=(16, 10))
        
        # 3D VCG
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(vx, vy, vz, 'b-', linewidth=2)
        ax1.scatter(vx[0], vy[0], vz[0], color='green', s=150, label='Start')
        ax1.scatter(vx[-1], vy[-1], vz[-1], color='red', s=150, label='End')
        ax1.set_xlabel('VX'); ax1.set_ylabel('VY'); ax1.set_zlabel('VZ')
        ax1.set_title('3D VCG Loop')
        ax1.legend()
        
        # Plane projections
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(vx, vy, 'r-', linewidth=2)
        ax2.scatter(vx[0], vy[0], color='green', s=100)
        ax2.scatter(vx[-1], vy[-1], color='red', s=100)
        ax2.set_xlabel('VX'); ax2.set_ylabel('VY')
        ax2.set_title('Frontal Plane (XY)')
        ax2.grid(True); ax2.axis('equal')
        
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(vx, vz, 'g-', linewidth=2)
        ax3.scatter(vx[0], vz[0], color='green', s=100)
        ax3.scatter(vx[-1], vz[-1], color='red', s=100)  
        ax3.set_xlabel('VX'); ax3.set_ylabel('VZ')
        ax3.set_title('Horizontal Plane (XZ)')
        ax3.grid(True); ax3.axis('equal')
        
        # Time series
        ax4 = fig.add_subplot(2, 3, 4)
        time_axis = np.arange(len(vx)) / 500
        ax4.plot(time_axis, vx, 'r-', label='VX', linewidth=2)
        ax4.plot(time_axis, vy, 'g-', label='VY', linewidth=2) 
        ax4.plot(time_axis, vz, 'b-', label='VZ', linewidth=2)
        ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Amplitude')
        ax4.set_title('VCG Components')
        ax4.legend(); ax4.grid(True)
        
        # Vector magnitude
        ax5 = fig.add_subplot(2, 3, 5)
        magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
        ax5.plot(time_axis, magnitude, 'purple', linewidth=2)
        ax5.set_xlabel('Time (s)'); ax5.set_ylabel('Vector Magnitude')
        ax5.set_title('VCG Magnitude')
        ax5.grid(True)
        
        # Synchronization summary
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
SYNCHRONIZATION SUMMARY

Leads synchronized: {successful_syncs}/{len(lead_names)}
Reference lead: {ref_lead.upper()}

Quality improvement:
• Overall: {overall_after - overall_before:+.3f}
• Limb leads: {limb_corr_after - limb_corr_before:+.3f}  
• Precordial: {prec_corr_after - prec_corr_before:+.3f}

Final correlations:
• Overall: {overall_after:.3f}
• Limb leads: {limb_corr_after:.3f}
• Precordial: {prec_corr_after:.3f}
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Robust Synchronized VCG Analysis', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        print("✅ VCG generated successfully!")
        
        # Individual heartbeat analysis
        print(f"\n=== INDIVIDUAL HEARTBEAT ANALYSIS ===")
        
        # Detect R-peaks in the reference lead for beat segmentation
        ref_idx = lead_names.index('ii') if 'ii' in lead_names else 0
        ref_lead_signal = synchronized_signals[:, ref_idx]
        r_peaks = detect_r_peaks_for_beats(ref_lead_signal, fs=500)
        
        print(f"Detected {len(r_peaks)} R-peaks for heartbeat segmentation")
        
        if len(r_peaks) >= 2:
            # Segment individual VCG beats
            individual_beats = segment_individual_beats(vcg_signals, r_peaks, fs=500)
            print(f"Successfully segmented {len(individual_beats)} individual VCG heartbeats")
            
            if len(individual_beats) > 0:
                # Plot each beat individually
                plot_individual_vcg_beats(individual_beats, ref_lead_signal, r_peaks, fs=500)
                
                # Calculate and plot average beat
                if len(individual_beats) > 1:
                    print(f"\nCalculating average heartbeat from {len(individual_beats)} beats...")
                    
                    # Find minimum length and calculate average
                    min_length = min(len(beat) for beat in individual_beats)
                    truncated_beats = [beat[:min_length] for beat in individual_beats]
                    avg_beat = np.mean(truncated_beats, axis=0)
                    
                    if avg_beat is not None:
                        # Plot average beat
                        vx_avg, vy_avg, vz_avg = avg_beat.T
                        
                        fig = plt.figure(figsize=(16, 10))
                        fig.suptitle(f'Average VCG Heartbeat (from {len(individual_beats)} beats)', fontsize=16, fontweight='bold')
                        
                        # 3D average
                        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
                        ax1.plot(vx_avg, vy_avg, vz_avg, 'b-', linewidth=4)
                        ax1.scatter(vx_avg[0], vy_avg[0], vz_avg[0], color='green', s=200, label='Start')
                        ax1.scatter(vx_avg[-1], vy_avg[-1], vz_avg[-1], color='red', s=200, label='End')
                        ax1.set_xlabel('VX'); ax1.set_ylabel('VY'); ax1.set_zlabel('VZ')
                        ax1.set_title('3D Average VCG Loop')
                        ax1.legend()
                        
                        # Plane projections
                        ax2 = fig.add_subplot(2, 3, 2)
                        ax2.plot(vx_avg, vy_avg, 'r-', linewidth=4)
                        ax2.scatter(vx_avg[0], vy_avg[0], color='green', s=150)
                        ax2.scatter(vx_avg[-1], vy_avg[-1], color='red', s=150)
                        ax2.set_xlabel('VX'); ax2.set_ylabel('VY')
                        ax2.set_title('Average Frontal Plane')
                        ax2.grid(True); ax2.axis('equal')
                        
                        ax3 = fig.add_subplot(2, 3, 3)
                        ax3.plot(vx_avg, vz_avg, 'g-', linewidth=4)
                        ax3.scatter(vx_avg[0], vz_avg[0], color='green', s=150)
                        ax3.scatter(vx_avg[-1], vz_avg[-1], color='red', s=150)
                        ax3.set_xlabel('VX'); ax3.set_ylabel('VZ')
                        ax3.set_title('Average Horizontal Plane')
                        ax3.grid(True); ax3.axis('equal')
                        
                        # Time series
                        ax4 = fig.add_subplot(2, 3, 4)
                        time_avg = np.arange(len(vx_avg)) / 500 * 1000
                        ax4.plot(time_avg, vx_avg, 'r-', label='VX', linewidth=3)
                        ax4.plot(time_avg, vy_avg, 'g-', label='VY', linewidth=3)
                        ax4.plot(time_avg, vz_avg, 'b-', label='VZ', linewidth=3)
                        ax4.set_xlabel('Time (ms)'); ax4.set_ylabel('Amplitude')
                        ax4.set_title('Average VCG Components')
                        ax4.legend(); ax4.grid(True)
                        
                        # Vector magnitude
                        ax5 = fig.add_subplot(2, 3, 5)
                        mag_avg = np.sqrt(vx_avg**2 + vy_avg**2 + vz_avg**2)
                        ax5.plot(time_avg, mag_avg, 'purple', linewidth=4)
                        ax5.set_xlabel('Time (ms)'); ax5.set_ylabel('Vector Magnitude')
                        ax5.set_title('Average Vector Magnitude')
                        ax5.grid(True)
                        
                        # Beat statistics
                        ax6 = fig.add_subplot(2, 3, 6)
                        ax6.axis('off')
                        
                        avg_stats_text = f"""
AVERAGE BEAT STATISTICS
(From {len(individual_beats)} heartbeats)

Max Vector Magnitude: {np.max(mag_avg):.1f} mV

Peak Amplitudes:
• VX: {np.max(np.abs(vx_avg)):.1f} mV
• VY: {np.max(np.abs(vy_avg)):.1f} mV
• VZ: {np.max(np.abs(vz_avg)):.1f} mV

Beat Duration: {time_avg[-1]:.1f} ms

Heart Rate: {60000/np.mean(np.diff(r_peaks))*500/500:.1f} BPM
                        """
                        
                        ax6.text(0.1, 0.9, avg_stats_text, transform=ax6.transAxes, 
                                fontsize=11, verticalalignment='top', fontfamily='monospace',
                                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                        
                        plt.tight_layout()
                        plt.show()
                        
                        print("✅ Average heartbeat analysis complete!")
                        
            else:
                print("⚠️ No complete heartbeats found for individual analysis")
        else:
            print("⚠️ Insufficient R-peaks detected for heartbeat segmentation")
        
        return {
            'synchronized_signals': synchronized_signals,
            'vcg_record': vcg_record,
            'sync_results': sync_results,
            'quality_improvement': overall_after - overall_before,
            'successful_syncs': successful_syncs
        }
        
    except Exception as e:
        print(f"❌ KORS VCG generation failed: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        
        return {
            'synchronized_signals': synchronized_signals,
            'vcg_record': None,
            'sync_results': sync_results,
            'quality_improvement': overall_after - overall_before,
            'successful_syncs': successful_syncs,
            'error': str(e)
        }

# Run the robust analysis
if __name__ == "__main__":
    print("Starting Robust ECG Synchronization for Scanned Data...")
    print("=" * 70)
    
    try:
        results = robust_ecg_synchronization()
        if results and results.get('vcg_record') is not None:
            print(f"\n" + "=" * 70)
            print("✅ ROBUST SYNCHRONIZATION COMPLETED!")
            print(f"✅ Successfully synchronized: {results['successful_syncs']}/12 leads")
            print(f"✅ Quality improvement: {results['quality_improvement']:+.3f}")
            print(f"✅ VCG generated from properly aligned data")
            print("=" * 70)
        elif results:
            print(f"\n" + "=" * 70)
            print("⚠️ SYNCHRONIZATION COMPLETED (VCG generation failed)")
            print(f"✅ Successfully synchronized: {results['successful_syncs']}/12 leads")
            print(f"✅ Quality improvement: {results['quality_improvement']:+.3f}")
            print(f"❌ VCG generation error: {results.get('error', 'Unknown error')}")
            print("=" * 70)
        else:
            print("❌ Analysis failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()