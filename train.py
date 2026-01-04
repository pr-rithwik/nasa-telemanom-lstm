"""Training loop for Telemanom LSTM"""
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm
import config
from model import TelemanomLSTM
from data_loader import get_dataloaders

class Trainer:
    """Handle training and validation"""
    
    def __init__(self):
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = TelemanomLSTM().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create checkpoint dir
        config.CHECKPOINT_DIR.mkdir(exist_ok=True)
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader):
        """Full training loop with early stopping"""
        print(f"\nTraining Telemanom LSTM for {config.EPOCHS} epochs...")
        
        for epoch in range(config.EPOCHS):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}/{config.EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint()
            else:
                self.patience_counter += 1
                if self.patience_counter >= config.PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"\nBest validation loss: {self.best_val_loss:.6f}")
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }, checkpoint_path)
    
    def load_checkpoint(self):
        """Load model checkpoint"""
        checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model with val loss: {checkpoint['best_val_loss']:.6f}")
        else:
            print("No checkpoint found")

def main():
    # Load data
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # Train model
    trainer = Trainer()
    trainer.train(train_loader, val_loader)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()