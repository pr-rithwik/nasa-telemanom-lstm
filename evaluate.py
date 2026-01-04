"""Evaluation and anomaly detection"""
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import config
from model import TelemanomLSTM
from data_loader import get_dataloaders
from train import Trainer

class Evaluator:
    """Evaluate model and detect anomalies"""
    
    def __init__(self):
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.model = TelemanomLSTM().to(self.device)
        
        # Load trained model
        trainer = Trainer()
        trainer.load_checkpoint()
        self.model = trainer.model
        self.model.eval()
        
    def get_reconstruction_errors(self, dataloader):
        """Get reconstruction errors for all samples"""
        errors = []
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                error = self.model.get_reconstruction_error(inputs)
                errors.append(error.cpu().numpy())
        
        return np.concatenate(errors)
    
    def compute_threshold(self, train_errors):
        """Compute anomaly threshold using percentile method"""
        threshold = np.percentile(train_errors, config.THRESHOLD_PERCENTILE)
        return threshold
    
    def evaluate(self, train_loader, val_loader, test_loader):
        """Full evaluation pipeline"""
        print("\nEvaluating model...")
        
        # Get reconstruction errors
        train_errors = self.get_reconstruction_errors(train_loader)
        val_errors = self.get_reconstruction_errors(val_loader)
        test_errors = self.get_reconstruction_errors(test_loader)
        
        # Compute threshold from training data
        threshold = self.compute_threshold(train_errors)
        print(f"Anomaly threshold: {threshold:.6f}")
        
        # Create labels (test data contains anomalies)
        # Assume last 20% of test data is failure
        test_labels = np.zeros(len(test_errors))
        anomaly_start = int(len(test_errors) * 0.8)
        test_labels[anomaly_start:] = 1
        
        # Predict anomalies
        predictions = (test_errors > threshold).astype(int)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predictions, average='binary', zero_division=0
        )
        
        try:
            roc_auc = roc_auc_score(test_labels, test_errors)
        except:
            roc_auc = 0.0
        
        # Print results
        print("\n" + "="*50)
        print("ANOMALY DETECTION RESULTS")
        print("="*50)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print("="*50)
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        print(f"\nConfusion Matrix:")
        print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # Save results
        self.save_results(train_errors, val_errors, test_errors, threshold, test_labels, predictions)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
    
    def save_results(self, train_errors, val_errors, test_errors, threshold, labels, predictions):
        """Save visualizations"""
        config.RESULTS_DIR.mkdir(exist_ok=True)
        
        # Plot 1: Error distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.hist(train_errors, bins=50, alpha=0.7, label='Train')
        plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('Training Error Distribution')
        plt.legend()
        
        plt.subplot(132)
        plt.hist(val_errors, bins=50, alpha=0.7, label='Validation', color='orange')
        plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
        plt.xlabel('Reconstruction Error')
        plt.title('Validation Error Distribution')
        plt.legend()
        
        plt.subplot(133)
        plt.hist(test_errors[labels==0], bins=50, alpha=0.7, label='Normal', color='green')
        plt.hist(test_errors[labels==1], bins=50, alpha=0.7, label='Anomaly', color='red')
        plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
        plt.xlabel('Reconstruction Error')
        plt.title('Test Error Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(config.RESULTS_DIR / 'error_distributions.png', dpi=150)
        print(f"\nSaved: {config.RESULTS_DIR / 'error_distributions.png'}")
        
        # Plot 2: Anomaly detection over time
        plt.figure(figsize=(12, 4))
        plt.plot(test_errors, label='Reconstruction Error', alpha=0.7)
        plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
        plt.fill_between(range(len(labels)), 0, test_errors.max(), 
                        where=labels==1, alpha=0.3, color='red', label='True Anomaly')
        plt.xlabel('Sample Index')
        plt.ylabel('Reconstruction Error')
        plt.title('Anomaly Detection Over Time')
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.RESULTS_DIR / 'anomaly_detection.png', dpi=150)
        print(f"Saved: {config.RESULTS_DIR / 'anomaly_detection.png'}")

def main():
    # Load data
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # Evaluate
    evaluator = Evaluator()
    results = evaluator.evaluate(train_loader, val_loader, test_loader)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()