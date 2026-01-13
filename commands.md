# Command Reference - LSTM Anomaly Detection

**Quick reference for running project scripts. All commands run from project root directory.**

---

## Initial Setup

### Clone Repository
```bash
git clone https://github.com/pr-rithwik/nasa-telemanom-lstm
cd nasa-telemanom-lstm
```
**Does:** Downloads project to your machine.

### Install Dependencies
```bash
pip install -r requirements.txt
```
**Does:** Installs PyTorch, NumPy, Matplotlib, scikit-learn, and other required packages.
*Note:* pytorch has to be installed separaltely as it should be aligned to the CUDA version of the system. The command to install it is in the comment section of `requirements.txt` file.
---

## Data Analysis (Optional but Recommended)

### Analyze Data Splits
```bash
python analyze_splits.py
```
**Does:** Analyzes bearing data to identify optimal train/validation/test split boundaries based on degradation patterns.  
**Result:** Creates `results/data_analysis.png` showing recommended split points and prints suggested TRAIN_SPLIT and VAL_SPLIT values for `config.py`.

---

## Training & Evaluation

### Train Model
```bash
python train.py
```
**Does:** Trains LSTM model on bearing data using GPU acceleration.  
**Result:** Saved model checkpoint in `checkpoints/best_model.pth`.

### Evaluate Model
```bash
python evaluate.py
```
**Does:** Tests trained model on test dataset, calculates metrics (F1, ROC-AUC, confusion matrix).  
**Result:** Prints metrics to console and saves `results/evaluation_results.png`.

### Run Complete Pipeline
```bash
python run_all.py
```
**Does:** Runs training followed by evaluation in sequence.  
**Note:** Skips the optional data analysis step (`analyze_splits.py`). If you want to optimize splits first, run that separately before `run_all.py`.

---

## Demonstrations

### Demo: Model on Real Test Data
```bash
python demo_model.py
```
**Does:** Loads trained model and runs inference on actual test data (files 952-984, real bearing failures). Shows detection statistics and error distributions.  
**Result:** Saves `results/demo_detection.png` showing anomalies detected in real bearing failure data.

### Demo: Normal vs Corrupted Data
```bash
python demo_corrupt.py
```
**Does:** Demonstrates model's ability to detect anomalies by comparing reconstruction errors on normal data vs artificially corrupted data (noise, spikes, drift, zeros).  
**Result:** Saves `results/corruption_demo.png` showing error distributions proving model can distinguish good from bad data.

---

## Workflow Examples

### First-Time Setup
```bash
git clone https://github.com/pr-rithwik/nasa-telemanom-lstm
cd telemanom_lstm
pip install -r requirements.txt
python analyze_splits.py        # Find optimal splits (optional)
# Update config.py with recommended splits if needed
python run_all.py               # Train + evaluate
python demo_model.py            # See results on real test data
python demo_corrupt.py          # Verify model works
```

### After Config Changes
```bash
python train.py                 # Retrain
python evaluate.py              # Re-evaluate
```

### Quick Check
```bash
python run_all.py               # Train + evaluate in one command
```

---

## Output Locations

**Models:**
- `checkpoints/best_model.pth` - Trained model weights

**Results:**
- `results/data_analysis.png` - Data split analysis
- `results/evaluation_results.png` - Test metrics and confusion matrix
- `results/demo_detection.png` - Real test data anomaly detection
- `results/corruption_demo.png` - Normal vs corrupt data comparison
- `results/training_loss.png` - Training history

**Logs:**
- Console output shows epoch-by-epoch training progress and final metrics

---

## Notes

- All scripts check for data availability before running
- GPU is used automatically if available (CUDA)
- Model checkpoints save automatically during training
- `run_all.py` does NOT run data analysis - run `analyze_splits.py` separately if needed
- Results overwrite previous outputs in `results/` folder
- Data files should already be in `data/2nd_test/` directory
