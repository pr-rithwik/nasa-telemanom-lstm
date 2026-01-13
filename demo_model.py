"""
Simple demo script - Shows trained model detecting anomalies
Run after training completes
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import config
from model import TelemanomLSTM
from train import Trainer
from data_loader import get_dataloaders

def demo_model():
    """Demonstrate trained model on test data"""
    
    print("="*60)
    print("TELEMANOM LSTM - ANOMALY DETECTION DEMO")
    print("="*60)
    
    # Load trained model
    print("\n1. Loading trained model...")
    trainer = Trainer()
    trainer.load_checkpoint()
    print("   ✓ Model loaded from checkpoint")
    
    # Load test data
    print("\n2. Loading test data (unseen during training)...")
    _, _, test_loader = get_dataloaders()
    print(f"   ✓ Test samples: {len(test_loader.dataset)}")
    
    # Get predictions
    print("\n3. Running inference on GPU...")
    import time
    start = time.time()
    
    errors = []
    trainer.model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(trainer.device)
            targets = targets.to(trainer.device)
            
            outputs = trainer.model(inputs)
            error = torch.mean((outputs - targets) ** 2, dim=1)
            errors.extend(error.cpu().numpy())
    
    end = time.time()
    errors = np.array(errors)
    
    print(f"   ✓ Processed {len(errors):,} samples in {end-start:.1f} seconds")
    print(f"   ✓ Speed: {len(errors)/(end-start):,.0f} samples/second")
    
    # Show statistics
    print("\n4. Anomaly Detection Results:")
    print(f"   Mean error: {errors.mean():.6f}")
    print(f"   Std error: {errors.std():.6f}")
    print(f"   Max error: {errors.max():.6f}")
    print(f"   Min error: {errors.min():.6f}")
    
    # Simple threshold
    threshold = np.percentile(errors, 95)
    anomalies = errors > threshold
    anomaly_count = anomalies.sum()
    anomaly_pct = 100 * anomaly_count / len(errors)
    
    print(f"\n5. Detection Summary:")
    print(f"   Threshold (95th percentile): {threshold:.6f}")
    print(f"   Anomalies detected: {anomaly_count:,} ({anomaly_pct:.1f}%)")
    print(f"   Normal samples: {len(errors) - anomaly_count:,}")
    
    # Quick visualization
    print("\n6. Creating quick visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Ensure errors and anomalies are 1D arrays of same length
    errors_flat = errors.flatten()
    anomalies_flat = anomalies.flatten()
    
    # Plot 1: Error over time
    ax = axes[0]
    ax.plot(errors_flat, alpha=0.7, linewidth=0.5, color='blue', label='Reconstruction Error')
    ax.axhline(threshold, color='r', linestyle='--', linewidth=2, label='Threshold')
    
    # Highlight anomaly regions
    anomaly_indices = np.where(anomalies_flat)[0]
    if len(anomaly_indices) > 0:
        ax.scatter(anomaly_indices, errors_flat[anomaly_indices], 
                  color='red', alpha=0.6, s=2, label=f'Anomalies ({len(anomaly_indices)})', zorder=5)
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Reconstruction Error')
    ax.set_title('Anomaly Detection Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Histogram
    ax = axes[1]
    ax.hist(errors_flat[~anomalies_flat], bins=50, alpha=0.7, label='Normal', color='blue')
    ax.hist(errors_flat[anomalies_flat], bins=50, alpha=0.7, label='Anomalies', color='red')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    demo_plot = config.RESULTS_DIR / "demo_detection.png"
    plt.savefig(demo_plot, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {demo_plot}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print(f"\nKey Takeaways:")
    print(f"  • Model successfully loaded from training")
    print(f"  • Processed {len(errors):,} samples in {end-start:.1f}s")
    print(f"  • Detected {anomaly_pct:.1f}% anomalies in test data")
    print(f"  • Ready for production use")
    print("\n" + "="*60)

if __name__ == "__main__":
    demo_model()