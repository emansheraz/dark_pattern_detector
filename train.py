import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.calibration import CalibratedClassifierCV

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pickle

# Check GPU availability
print("=" * 60)
print("DEVICE CHECK:")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 60)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
print("Loading dataset...")
df = pd.read_csv("dark_patterns_dataset_clean.csv", header=None, names=['text', 'label'])

# Drop empty rows
df = df.dropna()

print(f"✅ Dataset loaded: {len(df)} examples")
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution BEFORE encoding:")
print(df['label'].value_counts())
print()

# -----------------------------
# STEP 2: Encode Labels
# -----------------------------
print("Encoding labels...")
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Sorted alphabetically by sklearn: disguised_ad=0, fake_urgency=1, hidden_cost=2, normal=3
CLASS_NAMES = sorted(label_encoder.classes_.tolist())

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(f"\n🔍 LABEL MAPPING (index -> class name):")
print(f"   Alphabetical order (how sklearn encodes):")
for idx, class_name in enumerate(CLASS_NAMES):
    print(f"   {idx} -> '{class_name}'")
print(f"\nActual mapping used: {label_mapping}\n")

# Verify class distribution after encoding
print(f"Class distribution AFTER encoding:")
for class_name, idx in sorted(label_mapping.items()):
    count = (df['label'] == idx).sum()
    print(f"  {class_name:15s} (index {idx}): {count} examples")
print()

# -----------------------------
# STEP 3: Train-Test Split
# -----------------------------
print("Splitting data (80-20 stratified split)...")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"Training samples: {len(train_texts)}")
print(f"Testing samples: {len(test_texts)}\n")

# -----------------------------
# STEP 4: Load Tokenizer
# -----------------------------
print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize
print("Tokenizing texts...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

print("✅ Tokenization complete\n")

# -----------------------------
# Dataset Class
# -----------------------------
class DarkPatternDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = DarkPatternDataset(train_encodings, train_labels)
test_dataset = DarkPatternDataset(test_encodings, test_labels)

# -----------------------------
# STEP 5: Load Model
# -----------------------------
print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4
)

# Move model to GPU if available
model.to(device)
print(f"✅ Model loaded and moved to {device}\n")

# -----------------------------
# STEP 6: Training Arguments
# -----------------------------
print("Setting up training arguments...")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    logging_dir="./logs",
    save_strategy="epoch",
    report_to="none",
    logging_steps=10,
    seed=42,
    use_cpu=False if torch.cuda.is_available() else True
)

print(f"✅ Training will use: {device}\n")

# -----------------------------
# STEP 7: Evaluation Metrics (Per-Class)
# -----------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)

    # average=None returns a per-class array instead of a single weighted scalar
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )

    # HuggingFace Trainer requires a flat dict of scalar values — unpack each class
    metrics = {'accuracy': acc}
    for i, name in enumerate(CLASS_NAMES):
        metrics[f'precision_{name}'] = float(precision_per_class[i])
        metrics[f'recall_{name}']    = float(recall_per_class[i])
        metrics[f'f1_{name}']        = float(f1_per_class[i])

    return metrics

# -----------------------------
# STEP 8: Trainer
# -----------------------------
print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
print("\n" + "=" * 60)
print("STARTING TRAINING...")
print("=" * 60 + "\n")

trainer.train()

# -----------------------------
# EVALUATE
# -----------------------------
print("\n" + "=" * 60)
print("EVALUATING ON TEST SET...")
print("=" * 60 + "\n")

eval_results = trainer.evaluate()

# -----------------------------
# PRINT PER-CLASS RESULTS
# -----------------------------
print("\nFinal Results:")
print(f"  Accuracy: {eval_results.get('eval_accuracy', 0):.4f}\n")

col_w = 20
print(f"  {'Class':<{col_w}} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("  " + "-" * (col_w + 32))

for name in CLASS_NAMES:
    p = eval_results.get(f'eval_precision_{name}', 0)
    r = eval_results.get(f'eval_recall_{name}', 0)
    f = eval_results.get(f'eval_f1_{name}', 0)
    print(f"  {name:<{col_w}} {p:>10.4f} {r:>10.4f} {f:>10.4f}")

print()

# -----------------------------
# STEP 9: Save Model
# -----------------------------
print("Saving model...")

# Save model state_dict as .pt file
torch.save(model.state_dict(), "model/model.pt")
print("✅ Model saved to ./model/model.pt")

# Save tokenizer as .pkl file
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved to ./model/tokenizer.pkl")

# Save label encoder and mapping as .pkl file for inference
label_mapping_info = {
    'classes': label_encoder.classes_.tolist(),
    'label_mapping': label_mapping,
    'sorted_classes': CLASS_NAMES
}
with open("model/label_mapping.pkl", "wb") as f:
    pickle.dump(label_mapping_info, f)
print("✅ Label mapping saved to ./model/label_mapping.pkl")
print(f"\nSaved mapping info:")
print(f"  Classes: {label_mapping_info['classes']}")
print(f"  Sorted:  {label_mapping_info['sorted_classes']}")
print(f"  Mapping: {label_mapping_info['label_mapping']}")

print("\n✅ All models saved successfully!")