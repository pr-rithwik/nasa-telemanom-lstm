"""
Analyze bearing data to find optimal train/val/test splits
Run this BEFORE training to visualize when degradation actually starts
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def analyze_bearing_data():
    """Analyze all files to find degradation point"""
    
    # Path to data
    project_dir = Path(__file__).parent.absolute()
    data_dir = project_dir / "data" / "2nd_test"
    
    if not data_dir.exists():
        print("ERROR: Data not found!")
        print(f"Expected: {data_dir}")
        print("Run download_data.py first")
        return
    
    # Get all data files (no .txt extension)
    files = sorted([f for f in data_dir.iterdir() if f.is_file() and not f.name.startswith('.')])
    print(f"Found {len(files)} files\n")
    
    # Calculate statistics for each file
    file_stats = []
    
    print("Analyzing files...")
    for f in tqdm(files):
        data = np.loadtxt(f)
        
        # Calculate statistics
        stats = {
            'mean': data.mean(),
            'std': data.std(),
            'max': data.max(),
            'min': data.min(),
            'rms': np.sqrt((data ** 2).mean()),  # Root mean square
            'peak_to_peak': data.max() - data.min()
        }
        file_stats.append(stats)
    
    # Convert to arrays
    means = np.array([s['mean'] for s in file_stats])
    stds = np.array([s['std'] for s in file_stats])
    rms = np.array([s['rms'] for s in file_stats])
    peak_to_peak = np.array([s['peak_to_peak'] for s in file_stats])
    
    # Find degradation point (when RMS increases significantly)
    baseline_rms = np.median(rms[:100])  # First 100 files as baseline
    rms_threshold = baseline_rms * 1.5   # 50% increase
    
    degradation_start = None
    for i, r in enumerate(rms):
        if r > rms_threshold:
            degradation_start = i
            break
    
    # Find failure point (when RMS spikes dramatically)
    failure_threshold = baseline_rms * 3.0  # 200% increase
    failure_start = None
    for i, r in enumerate(rms):
        if r > failure_threshold:
            failure_start = i
            break
    
    # Plot analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: RMS over time
    ax = axes[0, 0]
    ax.plot(rms, label='RMS', alpha=0.7)
    ax.axhline(baseline_rms, color='g', linestyle='--', label='Baseline')
    ax.axhline(rms_threshold, color='orange', linestyle='--', label='Degradation Threshold')
    if degradation_start:
        ax.axvline(degradation_start, color='orange', linestyle='-', label=f'Degradation Start ({degradation_start})')
    if failure_start:
        ax.axvline(failure_start, color='r', linestyle='-', label=f'Failure Start ({failure_start})')
    ax.set_xlabel('File Index')
    ax.set_ylabel('RMS Value')
    ax.set_title('RMS Over Time (Key Indicator)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Standard deviation
    ax = axes[0, 1]
    ax.plot(stds, label='Std Dev', alpha=0.7, color='purple')
    if degradation_start:
        ax.axvline(degradation_start, color='orange', linestyle='--', alpha=0.5)
    if failure_start:
        ax.axvline(failure_start, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('File Index')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Vibration Variability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Peak-to-peak
    ax = axes[1, 0]
    ax.plot(peak_to_peak, label='Peak-to-Peak', alpha=0.7, color='green')
    if degradation_start:
        ax.axvline(degradation_start, color='orange', linestyle='--', alpha=0.5)
    if failure_start:
        ax.axvline(failure_start, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('File Index')
    ax.set_ylabel('Peak-to-Peak Amplitude')
    ax.set_title('Vibration Range')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: All metrics normalized
    ax = axes[1, 1]
    ax.plot(rms / rms.max(), label='RMS (normalized)', alpha=0.7)
    ax.plot(stds / stds.max(), label='Std Dev (normalized)', alpha=0.7)
    ax.plot(peak_to_peak / peak_to_peak.max(), label='Peak-to-Peak (normalized)', alpha=0.7)
    if degradation_start:
        ax.axvline(degradation_start, color='orange', linestyle='--', label=f'Degradation ({degradation_start})')
    if failure_start:
        ax.axvline(failure_start, color='r', linestyle='--', label=f'Failure ({failure_start})')
    ax.set_xlabel('File Index')
    ax.set_ylabel('Normalized Value')
    ax.set_title('All Metrics (Normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = project_dir / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "data_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved analysis plot: {output_path}")
    
    # Print recommendations
    print("\n" + "="*60)
    print("DATA ANALYSIS RESULTS")
    print("="*60)
    print(f"Total files: {len(files)}")
    print(f"Baseline RMS: {baseline_rms:.4f}")
    print(f"\nDetected degradation start: File {degradation_start if degradation_start else 'Not detected'}")
    print(f"Detected failure start: File {failure_start if failure_start else 'Not detected'}")
    
    print("\n" + "="*60)
    print("RECOMMENDED SPLITS")
    print("="*60)
    
    if degradation_start and failure_start:
        # Conservative: train only on clearly healthy data
        recommended_train = degradation_start - 50  # Buffer before degradation
        recommended_val = failure_start - 20        # Before failure
        
        print(f"\nOption 1: Conservative (maximize healthy training data)")
        print(f"  TRAIN_SPLIT = {recommended_train}")
        print(f"  VAL_SPLIT = {recommended_val}")
        print(f"  TEST = {recommended_val + 1}-{len(files)}")
        
        # Aggressive: use more data
        aggressive_train = degradation_start + 50
        aggressive_val = failure_start
        
        print(f"\nOption 2: Aggressive (include early degradation in training)")
        print(f"  TRAIN_SPLIT = {aggressive_train}")
        print(f"  VAL_SPLIT = {aggressive_val}")
        print(f"  TEST = {aggressive_val + 1}-{len(files)}")
        
        print(f"\n" + "="*60)
        print("COPY-PASTE READY VALUES")
        print("="*60)
        print(f"\n# Conservative (recommended)")
        print(f"TRAIN_SPLIT = {recommended_train}")
        print(f"VAL_SPLIT = {recommended_val}")
        
        print(f"\n# Aggressive")
        print(f"TRAIN_SPLIT = {aggressive_train}")
        print(f"VAL_SPLIT = {aggressive_val}")
                
    elif degradation_start and not failure_start:
        # Only degradation detected
        recommended_train = degradation_start - 50
        recommended_val = int(len(files) * 0.9)
        
        print(f"\nDegradation detected at file {degradation_start}")
        print(f"Failure point not clearly detected - using 90% for validation")
        
        print(f"\nRecommended:")
        print(f"  TRAIN_SPLIT = {recommended_train}")
        print(f"  VAL_SPLIT = {recommended_val}")
        print(f"  TEST = {recommended_val + 1}-{len(files)}")
        
        print(f"\n" + "="*60)
        print("COPY-PASTE READY VALUES")
        print("="*60)
        print(f"\nTRAIN_SPLIT = {recommended_train}")
        print(f"VAL_SPLIT = {recommended_val}")
        
    else:
        # Fallback to percentage-based
        fallback_train = int(len(files) * 0.7)
        fallback_val = int(len(files) * 0.9)
        
        print(f"\nNo clear degradation detected - using percentage-based:")
        print(f"  TRAIN_SPLIT = {fallback_train}  # 70%")
        print(f"  VAL_SPLIT = {fallback_val}   # 90%")
        print(f"  TEST = {fallback_val + 1}-{len(files)}")
        
        print(f"\n" + "="*60)
        print("COPY-PASTE READY VALUES")
        print("="*60)
        print(f"\nTRAIN_SPLIT = {fallback_train}")
        print(f"VAL_SPLIT = {fallback_val}")
    
    print("\n" + "="*60)
    print("CURRENT SETTINGS IN config.py")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Review the plot: results/data_analysis.png")
    print("2. Choose one of the recommended options above")
    print("3. Copy the values to config.py:")
    print("   - Open: config.py")
    print("   - Find: TRAIN_SPLIT and VAL_SPLIT")
    print("   - Replace with recommended values")
    print("4. Run: python run_all.py")
    
    if degradation_start and failure_start:
        print(f"\n💡 Quick recommendation: Use Conservative option")
        print(f"   Edit config.py and set:")
        print(f"   TRAIN_SPLIT = {degradation_start - 50}")
        print(f"   VAL_SPLIT = {failure_start - 20}")
    
    print("="*60)

if __name__ == "__main__":
    analyze_bearing_data()