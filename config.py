"""Configuration for Telemanom LSTM Anomaly Detection"""
from pathlib import Path

# Paths (dynamic - works on any machine)
PROJECT_DIR = Path(__file__).parent.absolute()  # Current folder
DATA_DIR = PROJECT_DIR / "data"                  # Data inside project
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"

# Data settings
TEST_NAME = "2nd_test"  # Using 2nd test dataset

# Train/Val/Test splits
# These are educated guesses - run analyze_splits.py to optimize!
# Current splits assume:
#   - Files 1-700: Healthy operation (71% of data)
#   - Files 701-884: Early degradation (19% of data)
#   - Files 885-984: Near failure (10% of data)
# 
# To find optimal splits:
#   1. Run: python analyze_splits.py
#   2. Review: results/data_analysis.png
#   3. Update these values based on recommendations
#   4. See SPLIT_SELECTION.md for detailed explanation
TRAIN_SPLIT = 857      # Files 1-700 for training (adjust after analysis)
VAL_SPLIT = 951        # Files 701-884 for validation (adjust after analysis)
                       # Files 885-984 for testing

# Model architecture (Telemanom paper specifications)
INPUT_DIM = 4          # 4 bearing channels
HIDDEN_DIM = 80        # LSTM hidden size (from paper)
LATENT_DIM = 40        # Bottleneck dimension
NUM_LAYERS = 2         # Stacked LSTM layers
DROPOUT = 0.3          # Dropout rate

# Training settings
SEQUENCE_LEN = 250     # Lookback window (from Telemanom paper)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
PATIENCE = 10          # Early stopping patience

# Anomaly detection
THRESHOLD_PERCENTILE = 95  # Threshold for anomaly (tune this)

# Hardware
DEVICE = "cuda"  # Will auto-fallback to CPU if no GPU