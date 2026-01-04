# Understanding Train/Val/Test Splits

## The Question

> "How did we decide on TRAIN_SPLIT = 700 and VAL_SPLIT = 884?"

**Short answer:** Educated guess based on standard practices.

**Better answer:** Run `analyze_splits.py` to find the actual degradation point in YOUR data.

---

## Why These Splits Matter

### Anomaly Detection Paradigm

**Key principle:** Train on NORMAL data only

```
Training data:   100% healthy
Validation data: Mix of healthy + early degradation
Test data:       Mix of degradation + failure
```

**Why?**
- Model learns "what normal looks like"
- Anomalies = anything that doesn't match normal
- If you train on failures, model thinks failures are normal!

---

## How We Chose Current Values

### The Data

**2nd_test dataset:**
- 984 files total
- Files every 10 minutes
- ~7 days of operation
- Failure at the END (we know from documentation)

### The Logic

```
Files 1-700 (71%):   Probably healthy
Files 701-884 (19%): Probably degrading
Files 885-984 (10%): Probably failing
```

**Sources of these numbers:**

1. **Standard ML split:** 70/15/15 or 80/10/10
2. **Anomaly papers:** Often use 60-80% for training
3. **Bearing research:** Degradation often starts in last 20-30% of life
4. **Conservative guess:** Better to have TOO MUCH healthy data than not enough

---

## The Problem with Guessing

**We don't actually know:**
- When degradation started (file 650? 750? 800?)
- How fast it progressed
- Whether there were multiple degradation phases

**Consequences:**

❌ **Train split too small** (e.g., 500)
- Not enough healthy data
- Model doesn't learn normal patterns well
- Higher false positives

❌ **Train split too large** (e.g., 850)
- Include degradation in training
- Model thinks degradation is normal
- Misses actual failures

✅ **Train split just right** (actual healthy data)
- Model learns pure normal behavior
- Detects degradation when it starts
- Good performance

---

## How to Find the Right Split

### Method 1: Run analyze_splits.py (Recommended)

```bash
python analyze_splits.py
```

**What it does:**
1. Loads all 984 files
2. Calculates RMS (root mean square) for each file
3. Finds when RMS increases significantly
4. Plots the progression
5. Recommends splits based on actual data

**Example output:**
```
Detected degradation start: File 723
Detected failure start: File 891

RECOMMENDED SPLITS:
  TRAIN_SPLIT = 673   # 50 files before degradation (buffer)
  VAL_SPLIT = 871     # 20 files before failure
  TEST = 872-984
```

### Method 2: Visual Inspection

1. Load a few files from different points:
```python
import numpy as np
import matplotlib.pyplot as plt

# Early (healthy)
early = np.loadtxt("data/2nd_test/2004.02.12.10.32.39")

# Middle (might be degrading)  
middle = np.loadtxt("data/2nd_test/2004.02.16.15.00.00")

# Late (probably failing)
late = np.loadtxt("data/2nd_test/2004.02.19.05.00.00")

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(131); plt.plot(early[:1000]); plt.title("Early")
plt.subplot(132); plt.plot(middle[:1000]); plt.title("Middle")
plt.subplot(133); plt.plot(late[:1000]); plt.title("Late")
plt.show()
```

2. Look for:
- Increased amplitude
- More noise/irregularity
- Sudden spikes
- Pattern changes

### Method 3: Literature

Check bearing failure research:
- Typical bearing life: 100M revolutions (dataset documentation)
- Degradation phase: Usually last 10-30% of life
- For 7-day test: Degradation likely starts day 5-6 (files 700-850)

---

## Current Split Analysis

### Our Current Choice

```python
TRAIN_SPLIT = 700      # Files 1-700 for training
VAL_SPLIT = 884        # Files 701-884 for validation
```

**Percentage breakdown:**
- Train: 700/984 = 71%
- Validation: 184/984 = 19%
- Test: 100/984 = 10%

**Pros:**
✅ Conservative (likely all healthy in training)
✅ Standard 70/20/10 split
✅ Gives model plenty of normal data
✅ Test set has clear failures

**Cons:**
❌ Might be leaving some healthy data unused
❌ Validation set might start too late
❌ Not optimized for THIS specific dataset

**Verdict:** Reasonable default, but can be improved with analysis

---

## What Different Splits Do

### Scenario 1: Train = 500 (Too Conservative)

```python
TRAIN_SPLIT = 500  # Only 50% of data
```

**Effect:**
- Less normal data to learn from
- Model might not capture all normal patterns
- Higher false positive rate
- Underutilized healthy data

### Scenario 2: Train = 900 (Too Aggressive)

```python
TRAIN_SPLIT = 900  # 91% of data
```

**Effect:**
- Probably includes degradation in training
- Model learns failures as "normal"
- Misses actual anomalies
- Poor performance

### Scenario 3: Train = 700 (Current)

```python
TRAIN_SPLIT = 700  # 71% of data
```

**Effect:**
- Good balance
- Likely all healthy data
- Standard split ratio
- Should work well

### Scenario 4: Train = Based on Analysis

```python
# After running analyze_splits.py
TRAIN_SPLIT = 673  # 50 files before detected degradation
```

**Effect:**
- Optimized for THIS dataset
- Maximum healthy data
- Clear separation
- Best performance (probably)

---

## Recommended Workflow

### First Time

1. **Use defaults** (TRAIN_SPLIT = 700)
2. **Train and evaluate**
3. **Check results**
4. **If results are poor** (<70% F1), analyze splits

### Optimization

1. **Run analysis:**
   ```bash
   python analyze_splits.py
   ```

2. **Review plot:**
   - Check `results/data_analysis.png`
   - See when RMS increases

3. **Update config.py:**
   ```python
   TRAIN_SPLIT = <recommended_value>
   VAL_SPLIT = <recommended_value>
   ```

4. **Retrain:**
   ```bash
   python run_all.py
   ```

5. **Compare results:**
   - Did F1 improve?
   - Fewer false positives?
   - Better detection timing?

---

## Rule of Thumb

**For bearing failure datasets:**

```python
# Conservative (safe default)
TRAIN_SPLIT = int(total_files * 0.70)  # 70%
VAL_SPLIT = int(total_files * 0.90)    # 90%

# After analysis (optimized)
TRAIN_SPLIT = degradation_point - 50   # 50-file buffer
VAL_SPLIT = failure_point - 20         # 20-file buffer
```

**General principle:**
- Train on definitely healthy data
- Validate on questionable data
- Test on definitely failing data

---

## Summary

### How We Chose 700/884

1. ✅ Standard 70/20/10 ML split
2. ✅ Conservative (better safe than sorry)
3. ✅ Based on bearing failure literature
4. ❌ NOT optimized for this specific dataset

### How YOU Should Choose

1. **Start with defaults** (700/884)
2. **Run `analyze_splits.py`**
3. **Look at the plot**
4. **Use recommended values**
5. **Compare performance**

### Bottom Line

**Current splits (700/884) are reasonable defaults that should work.**

**But for BEST results, run the analysis script and use data-driven splits.**

---

## Quick Reference

```bash
# Use defaults (good enough)
python run_all.py

# Optimize splits (better performance)
python analyze_splits.py  # See recommendations
# Update config.py with recommended values
python run_all.py         # Retrain with optimized splits
```

**The 5-10 minutes spent analyzing is worth it for 5-10% better F1 score!**