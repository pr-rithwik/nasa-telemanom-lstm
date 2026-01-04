"""Data loading and preprocessing for NASA IMS Bearing Data"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple
import config

class BearingDataset(Dataset):
    """Load IMS bearing data and create sequences"""
    
    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len]
        return seq, seq  # Autoencoder: input = target

def load_bearing_files(data_dir: Path, test_name: str, start: int, end: int) -> np.ndarray:
    """Load bearing data files from start to end index"""
    test_dir = data_dir / test_name
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Files might not have .txt extension, just get all files
    files = sorted([f for f in test_dir.iterdir() if f.is_file() and not f.name.startswith('.')])
    
    if not files:
        raise FileNotFoundError(f"No data files found in {test_dir}")
    
    # Load files in range
    data_list = []
    for f in files[start:end]:
        data = np.loadtxt(f)
        data_list.append(data)
    
    # Concatenate all files
    all_data = np.concatenate(data_list, axis=0)
    
    # Normalize (z-score)
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0) + 1e-8
    normalized = (all_data - mean) / std
    
    return normalized

def get_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders"""
    
    # Load data splits
    train_data = load_bearing_files(config.DATA_DIR, config.TEST_NAME, 0, config.TRAIN_SPLIT)
    val_data = load_bearing_files(config.DATA_DIR, config.TEST_NAME, config.TRAIN_SPLIT, config.VAL_SPLIT)
    test_data = load_bearing_files(config.DATA_DIR, config.TEST_NAME, config.VAL_SPLIT, None)
    
    # Create datasets
    train_dataset = BearingDataset(train_data, config.SEQUENCE_LEN)
    val_dataset = BearingDataset(val_data, config.SEQUENCE_LEN)
    test_dataset = BearingDataset(test_data, config.SEQUENCE_LEN)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader