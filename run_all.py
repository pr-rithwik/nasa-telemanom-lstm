"""
Master script for Telemanom LSTM Anomaly Detection
Run this after downloading the bearing data to the project's data/ folder
"""
import sys
from pathlib import Path

def check_data():
    """Check if data is available"""
    project_dir = Path(__file__).parent.absolute()
    data_path = project_dir / "data" / "2nd_test"
    
    if not data_path.exists():
        print("ERROR: Data not found!")
        print(f"Expected location: {data_path}")
        print("\nPlease run: python download_data.py")
        print("And follow instructions to download the bearing dataset")
        return False
    
    files = [f for f in data_path.iterdir() if f.is_file() and not f.name.startswith('.')]
    print(f"✓ Found {len(files)} data files at: {data_path}")
    return len(files) > 0

def main():
    print("="*60)
    print("NASA Telemanom LSTM - Bearing Anomaly Detection")
    print("="*60)
    
    # Check data
    if not check_data():
        sys.exit(1)
    
    # Train
    print("\n[1/2] Training model...")
    print("-"*60)
    from train import main as train_main
    train_main()
    
    # Evaluate
    print("\n[2/2] Evaluating model...")
    print("-"*60)
    from evaluate import main as eval_main
    eval_main()
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("\nResults saved to:")
    print("  - Model: checkpoints/best_model.pt")
    print("  - Plots: results/")
    print("\nNext steps:")
    print("  - Review results in results/ folder")
    print("  - Adjust THRESHOLD_PERCENTILE in config.py if needed")
    print("  - Present to manager!")

if __name__ == "__main__":
    main()