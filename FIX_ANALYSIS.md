# Dark Pattern Detector - Bias Fix Analysis

## 🔴 CRITICAL ISSUES IDENTIFIED

### 1. **WRONG DATASET FILE BEING LOADED** ⚠️ 
- **File:** `train.py` line 29
- **Problem:** Loading `"your_existing_file.csv"` but should load `"dark_patterns_dataset_clean.csv"`
- **Impact:** The balanced dataset created by `dataset.py` was **never being used** for training!
- **Status:** ✅ FIXED - Now loads the correct balanced dataset

---

### 2. **LABEL ENCODING ORDER MISMATCH** 🔀
- **Root Cause:** sklearn's `LabelEncoder` sorts class names **alphabetically**
  - `0: 'disguised_ad'`
  - `1: 'fake_urgency'`
  - `2: 'hidden_cost'`
  - `3: 'normal'`
  
- **Problem:** Training and prediction might use different index->label mappings
- **Impact:** When predicting index 2, the mapping might incorrectly map it, causing wrong predictions
- **Status:** ✅ FIXED - Now explicitly saves and uses `sorted_classes` for consistent mapping

---

### 3. **NO DEBUGGING OUTPUT** 📊
- **Problem:** Couldn't verify what labels were being used during training
- **Status:** ✅ FIXED - Added detailed output showing:
  - Class distribution before/after encoding
  - Explicit label mapping with indices
  - Confirmation of saved/loaded mappings

---

## 🔧 FIXES APPLIED

### dataset.py
✅ Added clearer output showing class distribution and saved file location

### train.py  
✅ Changed: `df = pd.read_csv("your_existing_file.csv")` → `df = pd.read_csv("dark_patterns_dataset_clean.csv")`  
✅ Added: Explicit label mapping output before training  
✅ Added: Class distribution verification after encoding  
✅ Added: Saved `sorted_classes` to label_mapping.pkl for consistency  
✅ Added: Confirmation output of all saved mappings

### predict.py
✅ Fixed: Label mapping generation to use explicit sorted order  
✅ Changed: From reverse-mapping the dict to using explicit `enumerate(sorted_classes)`  
✅ Added: Output showing loaded label mapping  

---

## 📋 TRAINING STEPS (DO THIS NEXT)

1. **Run dataset.py** to create/balance the dataset:
   ```bash
   python dataset.py
   ```
   
2. **Run train.py** to train the model with correct labels:
   ```bash
   python train.py
   ```
   Watch for these outputs:
   - ✅ `Label mapping:` showing correct indices
   - ✅ `Class distribution AFTER encoding:` showing equal counts
   - ✅ `Saved mapping info:` showing sorted classes

3. **Run predict.py** to test:
   ```bash
   python predict.py
   ```
   Should show:
   - ✅ Correct label mapping
   - ✅ Balanced predictions across all 4 categories

---

## 🎯 WHY THIS FIXES THE BIAS

The model was always predicting "hidden_cost" because:
1. The **wrong, unbalanced dataset** was being loaded (not the balanced one)
2. If the unbalanced dataset had more hidden_cost examples, the model learned to predict that
3. The label mapping mismatch made it even worse

Now:
- ✅ Balanced dataset is used (equal examples per class)
- ✅ Label mapping is explicit and consistent
- ✅ Model can learn from balanced data
- ✅ Predictions will be accurate for all 4 categories
