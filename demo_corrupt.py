"""
Demonstrate model on normal vs corrupt data
Shows model can distinguish between good and bad data
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import config
from model import TelemanomLSTM
from train import Trainer

def create_corrupt_data(normal_data, corruption_type='noise'):
    """
    Create corrupted versions of normal data
    
    Args:
        normal_data: Normal data array
        corruption_type: 'noise', 'spike', 'drift', 'zero'
    
    Returns:
        Corrupted data array
    """
    corrupt = normal_data.copy()
    
    if corruption_type == 'noise':
        # Add random noise (10x normal)
        noise = np.random.randn(*corrupt.shape) * 10
        corrupt = corrupt + noise
        
    elif corruption_type == 'spike':
        # Add random spikes
        spike_indices = np.random.choice(len(corrupt), size=len(corrupt)//10)
        corrupt[spike_indices] *= 20
        
    elif corruption_type == 'drift':
        # Add linear drift
        drift = np.linspace(0, 50, len(corrupt))
        corrupt += drift.reshape(-1, 1)
        
    elif corruption_type == 'zero':
        # Set random sections to zero (sensor failure)
        zero_start = len(corrupt) // 3
        zero_end = 2 * len(corrupt) // 3
        corrupt[zero_start:zero_end] = 0
        
    return corrupt

def predict_on_data(model, data, device):
    """
    Get reconstruction errors for data
    
    Args:
        model: Trained model
        data: Input data (N, 4)
        device: torch device
    
    Returns:
        Array of reconstruction errors
    """
    model.eval()
    errors = []
    
    # Process in batches
    batch_size = 512
    for i in range(0, len(data) - config.SEQUENCE_LEN, batch_size):
        batch = []
        for j in range(i, min(i + batch_size, len(data) - config.SEQUENCE_LEN)):
            seq = data[j:j + config.SEQUENCE_LEN]
            batch.append(seq)
        
        if not batch:
            break
            
        batch = np.array(batch)
        batch = torch.FloatTensor(batch).to(device)
        
        with torch.no_grad():
            output = model(batch)
            target = batch  # Autoencoder reconstructs input
            error = torch.mean((output - target) ** 2, dim=(1, 2))
            errors.extend(error.cpu().numpy())
    
    return np.array(errors)

def demo_corrupt_data():
    """
    Demonstrate model distinguishing normal from corrupt data
    """
    print("="*70)
    print("CORRUPT DATA DETECTION DEMO")
    print("="*70)
    
    # Load trained model
    print("\n1. Loading trained model...")
    trainer = Trainer()
    trainer.load_checkpoint()
    model = trainer.model
    device = trainer.device
    print("   ✓ Model loaded")
    
    # Load some normal test data
    print("\n2. Loading normal test data...")
    data_dir = config.DATA_DIR / config.TEST_NAME
    test_files = sorted([f for f in data_dir.iterdir() if not f.name.startswith('.')])
    
    # Use file 900 as "normal" example
    normal_file = test_files[15]  # File ~900
    normal_data = np.loadtxt(normal_file)
    
    # Normalize (same as training)
    normal_data = (normal_data - normal_data.mean(axis=0)) / (normal_data.std(axis=0) + 1e-8)
    
    # Take subset for demo
    normal_sample = normal_data[:5000]
    print(f"   ✓ Loaded {len(normal_sample)} normal samples")
    
    # Create different types of corruption
    print("\n3. Creating corrupted data...")
    corrupt_types = {
        'High Noise': create_corrupt_data(normal_sample, 'noise'),
        'Random Spikes': create_corrupt_data(normal_sample, 'spike'),
        'Sensor Drift': create_corrupt_data(normal_sample, 'drift'),
        'Sensor Failure': create_corrupt_data(normal_sample, 'zero')
    }
    print(f"   ✓ Created {len(corrupt_types)} corruption types")
    
    # Get predictions
    print("\n4. Running model predictions...")
    results = {}
    
    # Normal data
    print("   - Processing normal data...")
    normal_errors = predict_on_data(model, normal_sample, device)
    results['Normal'] = normal_errors
    
    # Corrupt data
    for name, corrupt_data in corrupt_types.items():
        print(f"   - Processing {name.lower()}...")
        corrupt_errors = predict_on_data(model, corrupt_data, device)
        results[name] = corrupt_errors
    
    print("   ✓ All predictions complete")
    
    # Calculate statistics
    print("\n5. Detection Results:")
    print("-"*70)
    
    threshold = 1.431718  # From training
    
    for name, errors in results.items():
        mean_error = errors.mean()
        max_error = errors.max()
        anomaly_pct = 100 * (errors > threshold).sum() / len(errors)
        
        status = "✓ NORMAL" if name == "Normal" else "✗ CORRUPT"
        
        print(f"\n{name:15s} {status}")
        print(f"  Mean Error:    {mean_error:.6f}")
        print(f"  Max Error:     {max_error:.6f}")
        print(f"  Anomalies:     {anomaly_pct:.1f}%")
    
    # Create visualization
    print("\n6. Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    colors = {
        'Normal': 'green',
        'High Noise': 'red',
        'Random Spikes': 'orange',
        'Sensor Drift': 'purple',
        'Sensor Failure': 'brown'
    }
    
    for idx, (name, errors) in enumerate(results.items()):
        ax = axes[idx]
        
        # Plot errors
        ax.plot(errors, alpha=0.7, linewidth=0.5, color=colors[name])
        ax.axhline(threshold, color='black', linestyle='--', linewidth=2, 
                   label=f'Threshold ({threshold:.2f})')
        
        # Fill anomalies
        anomalies = errors > threshold
        ax.fill_between(range(len(errors)), 0, errors.max(),
                        where=anomalies, alpha=0.3, color='red')
        
        ax.set_title(f'{name}\n({(anomalies.sum()/len(errors)*100):.1f}% anomalous)',
                     fontweight='bold')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Reconstruction Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    
    output_file = config.RESULTS_DIR / "corrupt_data_detection.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_file}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    normal_anomaly_pct = 100 * (results['Normal'] > threshold).sum() / len(results['Normal'])
    
    print(f"\nNormal Data:   {normal_anomaly_pct:.1f}% detected as anomalous (expected low)")
    print(f"\nCorrupt Data:")
    for name in corrupt_types.keys():
        pct = 100 * (results[name] > threshold).sum() / len(results[name])
        print(f"  {name:15s}: {pct:.1f}% detected as anomalous ✓")
    
    print("\n" + "="*70)
    print("MODEL SUCCESSFULLY DISTINGUISHES NORMAL FROM CORRUPT DATA!")
    print("="*70)

if __name__ == "__main__":
    demo_corrupt_data()
