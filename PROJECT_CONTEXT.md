# 🕵️‍♂️ Dark Pattern Detector - Project Context

## Project Overview

**Dark Pattern Detector** is an AI-powered system that automatically identifies and classifies deceptive e-commerce text using Natural Language Processing (NLP) and Transformer-based deep learning models. The system achieves **95.27% accuracy** in detecting manipulative user interface patterns that trick customers into making unintended purchases.

### Key Objectives
- Detect manipulative e-commerce text patterns
- Classify text into 4 distinct dark pattern categories
- Provide real-time prediction capabilities
- Handle imbalanced datasets effectively
- Leverage BERT (Bidirectional Encoder Representations from Transformers) for contextual understanding

---

## 📁 Project Folder Structure

```
dark_pattern_detector/
│
├── 📂 .git/                          # Git version control directory
├── 📂 .venv/                         # Python virtual environment (dependencies installed here)
├── 📂 .vscode/                       # VS Code configuration files
├── 📂 data/                          # Training datasets directory
│   ├── dataset.csv                   # Main training dataset (32 labeled examples)
│   └── dark_patterns_dataset_clean.csv (linked/reference)
│
├── 📂 model/                         # [Generated after training] Fine-tuned BERT model directory
│   ├── config.json                   # Model configuration
│   ├── pytorch_model.bin             # Trained model weights
│   ├── tokenizer.json                # BERT tokenizer config
│   └── vocab.txt                     # BERT vocabulary
│
├── 📂 results/                       # [Generated after training] Training results directory
│   └── checkpoint-*                  # Checkpoint files from training
│
├── 📂 logs/                          # [Generated after training] Training logs directory
│
├── 🐍 dataset.py                     # Data augmentation script
├── 🐍 train.py                       # Model training script
├── 🐍 predict.py                     # Inference/prediction script
├── 📄 requirements.txt                # Python dependencies
├── 📄 README.md                       # Project documentation
├── 📄 PROJECT_CONTEXT.md             # This file - comprehensive project context
├── 🔧 .gitignore                     # Git ignore rules
├── 📊 Screenshot 2026-04-24 154000.png # Project screenshot
├── 📊 your_existing_file.csv         # Augmented dataset output
└── 📊 dark_patterns_dataset_clean.csv # Base dataset for augmentation
```

---

## 📋 File Details

### Core Python Scripts

#### **train.py** - Model Training Pipeline
**Purpose:** Trains the fine-tuned BERT model on dark pattern data

**Key Steps:**
1. Loads dataset from `data/dataset.csv`
2. Preprocesses text and encodes labels (0-3 for 4 classes)
3. Tokenizes text using BERT tokenizer (max length: 512 tokens)
4. Splits data into 80% training and 20% testing (stratified)
5. Fine-tunes `bert-base-uncased` model for 3 epochs
6. Batch size: 8 (training and evaluation)
7. Evaluates using accuracy, precision, recall, F1 score
8. Saves trained model and tokenizer to `model/` directory

**Input File:** `data/dataset.csv`  
**Output Files:** `model/` directory  
**Dependencies:** pandas, torch, transformers, scikit-learn

---

#### **predict.py** - Inference Script
**Purpose:** Uses trained model to classify new e-commerce text

**Features:**
- Loads pre-trained BERT model from `model/`
- Classifies text into 4 categories
- Max token length: 64 characters
- Returns predicted class label

**Label Mapping:**
- `0` → `disguised_ad`
- `1` → `fake_urgency`
- `2` → `hidden_cost`
- `3` → `normal`

**Example Output:**
```python
predict("Extra charges will be added at checkout")
# Returns: "hidden_cost"
```

**Input:** User-provided text string  
**Output:** Classification label  
**Dependencies:** torch, transformers

---

#### **dataset.py** - Data Augmentation Script
**Purpose:** Expands training dataset with 400+ new examples

**Workflow:**
1. Loads existing dataset from `dark_patterns_dataset_clean.csv`
2. Adds 400+ hidden cost examples (various deceptive payment phrases)
3. Adds 15 normal/legitimate examples
4. Removes duplicate entries
5. Saves augmented dataset to `your_existing_file.csv`

**Output Statistics:**
- Original dataset: ~20 examples
- New hidden cost examples: 400+
- New normal examples: 15
- Final deduplicated dataset: ~430+ examples

**Purpose of Augmentation:** 
- Balances the dataset (hidden_cost class was underrepresented)
- Increases training data diversity
- Improves model robustness

**Input File:** `dark_patterns_dataset_clean.csv`  
**Output File:** `your_existing_file.csv`  
**Dependencies:** pandas

---

### Configuration & Dependencies

#### **requirements.txt** - Python Dependencies
```
pandas              # Data manipulation and CSV handling
scikit-learn        # Machine learning utilities, label encoding, metrics
transformers        # Hugging Face BERT models and tokenizers
torch               # PyTorch deep learning framework
```

**Installation:**
```powershell
pip install -r requirements.txt
```

---

### Documentation Files

#### **README.md** - Project Documentation
Contains:
- Project overview and key features
- Dark pattern categories with examples
- Dataset distribution table
- AI architecture diagram (text-based)
- Technical workflow explanation
- Feature highlights

#### **PROJECT_CONTEXT.md** - This File
Comprehensive project documentation including:
- Folder structure
- File descriptions
- Execution workflow
- Data pipeline
- Model architecture
- Training specifications

---

## 🏗️ Data Pipeline Architecture

```
Raw Text Input
     ↓
Preprocessing (cleaning, lowercasing)
     ↓
BERT Tokenizer
     ├─ Tokenize text into subword tokens
     ├─ Generate attention masks
     └─ Pad/truncate to max_length=512
     ↓
Fine-Tuned BERT Model
     ├─ Bidirectional encoding
     ├─ Contextual embeddings
     └─ Transfer learning from bert-base-uncased
     ↓
Classification Head
     ├─ Linear layer
     ├─ Dropout regularization
     └─ Softmax for 4-class prediction
     ↓
Output: [normal | fake_urgency | hidden_cost | disguised_ad]
```

---

## 📊 Dark Pattern Categories

| Category | Description | Examples | Dataset Size |
|----------|-------------|----------|---------------|
| **Normal** | Legitimate, transparent text | "Free delivery on orders above $50" | 1,355 |
| **Fake Urgency** | Creates artificial scarcity/pressure | "Only 2 items left!", "Sale ends in 24 hours" | 874 |
| **Hidden Cost** | Reveals charges late in checkout | "Extra fees at payment", "Final price at checkout" | 398+ |
| **Disguised Ad** | Ads presented as organic content | "Sponsored products you may like", "Recommended for you" | 226 |

---

## 🔄 Execution Workflow

### Step 1: Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2: Data Augmentation (Optional but Recommended)
```powershell
python dataset.py
```
- Expands training dataset with 400+ examples
- Reduces class imbalance
- Generates `your_existing_file.csv`

### Step 3: Train the Model
```powershell
python train.py
```
- Trains BERT on dataset
- Creates `model/` directory with weights and tokenizer
- Outputs: accuracy, precision, recall, F1 metrics
- Expected accuracy: ~95%

### Step 4: Make Predictions
```powershell
python predict.py
```
- Uses trained model to classify text
- Tests with sample inputs
- Demonstrates inference capability

---

## 🧠 Model Architecture Details

### Base Model: BERT (bert-base-uncased)
- **Type:** Transformer-based language model
- **Parameters:** 110M
- **Training Data:** English Wikipedia + BookCorpus
- **Tokenizer:** WordPiece (30,522 vocabulary)

### Fine-Tuning Specifications
- **Number of Labels:** 4 (multi-class classification)
- **Training Epochs:** 3
- **Batch Size:** 8 (training and evaluation)
- **Learning Rate:** Default (2e-5)
- **Max Sequence Length:** 512 tokens
- **Evaluation Strategy:** Per epoch
- **Save Strategy:** Per epoch (checkpoints)

### Classification Head
```
BERT Output Embeddings [CLS] Token (768-dim)
     ↓
Dropout Layer (probability: 0.1)
     ↓
Linear Layer (768 → 4)
     ↓
Softmax Activation
     ↓
Class Probabilities
```

---

## 📈 Performance Metrics

### Expected Results (from README)
- **Accuracy:** 95.27%
- **Precision:** Weighted average across classes
- **Recall:** Handles imbalanced dataset effectively
- **F1-Score:** Balanced performance metric

### Evaluation Method
- **Train-Test Split:** 80-20 (stratified sampling)
- **Stratification:** Preserves class distribution
- **Metric Calculation:** Weighted by class support

---

## 🔧 Key Features & Highlights

✅ **Transformer-Based NLP**
- Uses bidirectional BERT for contextual understanding
- Better than bag-of-words or keyword matching

✅ **Multi-Class Classification**
- Robust categorization into 4 distinct patterns
- Handles imbalanced dataset with stratified sampling

✅ **Real-Time Prediction**
- Fast inference with pre-trained model
- Can classify text instantly

✅ **Data Augmentation**
- 400+ additional examples included
- Improves model generalization

✅ **Transfer Learning**
- Leverages pre-trained BERT weights
- Requires fewer training examples

✅ **Comprehensive Evaluation**
- Multiple metrics: accuracy, precision, recall, F1
- Weighted metrics for imbalanced data

---

## 📁 Data Files Reference

| File | Purpose | Source |
|------|---------|--------|
| `data/dataset.csv` | Main training dataset | Created (32 initial examples) |
| `dark_patterns_dataset_clean.csv` | Base dataset for augmentation | Reference file |
| `your_existing_file.csv` | Augmented dataset output | Generated by `dataset.py` |
| `model/` | Fine-tuned BERT weights | Generated by `train.py` |
| `results/` | Training checkpoints | Generated by `train.py` |
| `logs/` | Training logs | Generated by `train.py` |

---

## ⚙️ System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2GB disk space (for model)
- CUDA GPU (optional but recommended for faster training)

**Recommended:**
- Python 3.10+
- 8GB+ RAM
- GPU with CUDA support
- SSD storage

---

## 🚀 Quick Start Commands

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Augment dataset (optional)
python dataset.py

# 4. Train model
python train.py

# 5. Test predictions
python predict.py
```

---

## 📝 Project Status & Next Steps

**Current State:**
- ✅ BERT model architecture implemented
- ✅ Training pipeline complete
- ✅ Inference script ready
- ✅ Data augmentation script included
- ✅ 95.27% accuracy achieved

**Potential Improvements:**
- Expand training dataset further
- Fine-tune hyperparameters
- Experiment with other models (RoBERTa, DistilBERT)
- Deploy as API endpoint
- Create web interface for testing
- Add confidence scores to predictions
- Implement batch prediction

---

## 📚 Technology Stack

| Component | Technology |
|-----------|-----------|
| **ML Framework** | PyTorch |
| **NLP Library** | Hugging Face Transformers |
| **Language Model** | BERT (bert-base-uncased) |
| **Data Processing** | Pandas, Scikit-learn |
| **Programming Language** | Python 3.8+ |

---

## 🎯 Use Cases

1. **E-Commerce Moderation** - Detect deceptive product descriptions
2. **Content Compliance** - Flag manipulative marketing text
3. **User Protection** - Warn users about dark patterns
4. **A/B Testing** - Validate if UI copy uses dark patterns
5. **Research** - Study deceptive practices in online retail

---

*Last Updated: May 23, 2026*
*Project: Dark Pattern Detector AI System*
