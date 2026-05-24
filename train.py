import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
 
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import pickle
 
# ============================================================
# DEVICE CHECK
# ============================================================
print("=" * 60)
print("DEVICE CHECK:")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name:   {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 60)
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")
 
# ============================================================
# STEP 1: Load Dataset
# ============================================================
print("Loading dataset...")
df = pd.read_csv("dark_patterns_dataset_clean.csv")[['text', 'label']].dropna()
 
print(f"✅ Dataset loaded: {len(df)} examples")
print(f"\nClass distribution BEFORE encoding:")
print(df['label'].value_counts())
print()
 
# ============================================================
# STEP 2: Encode Labels
# ============================================================
print("Encoding labels...")
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
 
# sklearn sorts alphabetically:
# 0 = disguised_ad | 1 = fake_urgency | 2 = hidden_cost | 3 = normal
CLASS_NAMES = sorted(label_encoder.classes_.tolist())
NUM_CLASSES  = len(CLASS_NAMES)
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
 
print(f"\n🔍 LABEL MAPPING (sorted alphabetically by sklearn):")
for idx, name in enumerate(CLASS_NAMES):
    count = (df['label'] == idx).sum()
    print(f"   {idx} -> '{name}'  ({count} examples)")
print()
 
# ============================================================
# STEP 3: Train / Test Split  (stratified 80-20)
# ============================================================
print("Splitting data (80-20 stratified split)...")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)
print(f"Training samples: {len(train_texts)}")
print(f"Testing  samples: {len(test_texts)}\n")
 
# ============================================================
# STEP 4: Tokenise
# ============================================================
print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
 
print("Tokenizing texts...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings  = tokenizer(test_texts,  truncation=True, padding=True, max_length=128)
print("✅ Tokenization complete\n")
 
# ============================================================
# Dataset Class
# ============================================================
class DarkPatternDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels
 
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
 
    def __len__(self):
        return len(self.labels)
 
train_dataset = DarkPatternDataset(train_encodings, train_labels)
test_dataset  = DarkPatternDataset(test_encodings,  test_labels)
 
# ============================================================
# STEP 5: Load Model
# ============================================================
print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=NUM_CLASSES
)
model.to(device)
print(f"✅ Model loaded and moved to {device}\n")
 
# ============================================================
# STEP 6: Training Arguments
#   — load_best_model_at_end  : keeps the checkpoint with best macro-F1
#   — metric_for_best_model   : uses macro F1 (not accuracy / weighted)
#   — warmup_ratio            : 10 % warmup helps BERT fine-tuning
#   — weight_decay            : L2 regularisation to reduce overfitting
#   — learning_rate 2e-5      : standard sweet-spot for BERT fine-tuning
# ============================================================
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir                  = "./results",
    num_train_epochs            = 5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    logging_dir                 = "./logs",
    logging_steps               = 10,
    report_to                   = "none",
    seed                        = 42,
 
    # ── Optimiser settings ──────────────────────────────────
    learning_rate               = 2e-5,   # best for BERT fine-tuning
    warmup_ratio                = 0.1,    # 10% of steps for LR warm-up
    weight_decay                = 0.01,   # L2 regularisation
 
    # ── Best-model selection based on macro F1 ──────────────
    load_best_model_at_end      = True,
    metric_for_best_model       = "f1_macro",
    greater_is_better           = True,
 
    use_cpu = not torch.cuda.is_available()
)
print(f"✅ Training configuration ready\n")
 
# ============================================================
# STEP 7: compute_metrics  — MACRO average (no class weighting)
#   average='macro' : every class contributes equally to the score
#                     regardless of how many samples it has.
#   average=None    : returns a per-class array so we can log
#                     individual class scores alongside the macro score.
# ============================================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
 
    acc = accuracy_score(labels, preds)
 
    # ── Macro scores (equal weight per class) ───────────────
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
 
    # ── Per-class scores (no averaging) ─────────────────────
    p_per, r_per, f1_per, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
 
    metrics = {
        'accuracy' : acc,
        'precision_macro': float(p_macro),
        'recall_macro'   : float(r_macro),
        'f1_macro'       : float(f1_macro),   # ← used for best-model selection
    }
 
    # Per-class breakdown
    for i, name in enumerate(CLASS_NAMES):
        metrics[f'precision_{name}'] = float(p_per[i])
        metrics[f'recall_{name}']    = float(r_per[i])
        metrics[f'f1_{name}']        = float(f1_per[i])
 
    return metrics
 
# ============================================================
# STEP 8: Trainer
# ============================================================
print("Initializing trainer...")
trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = test_dataset,
    compute_metrics = compute_metrics,
)
 
# ============================================================
# TRAIN
# ============================================================
print("\n" + "=" * 60)
print("STARTING TRAINING...")
print("=" * 60 + "\n")
 
trainer.train()
 
# ============================================================
# EVALUATE
# ============================================================
print("\n" + "=" * 60)
print("EVALUATING ON TEST SET...")
print("=" * 60 + "\n")
 
eval_results = trainer.evaluate()
 
# ============================================================
# PRINT RESULTS — per-class + macro summary
# ============================================================
W = 20
SEP = "  " + "-" * (W + 34)
 
print("\n📊 FINAL EVALUATION RESULTS")
print("=" * 60)
print(f"\n  Overall Accuracy : {eval_results.get('eval_accuracy', 0):.4f}")
print(f"\n  Macro Summary (equal weight per class):")
print(f"    Macro Precision : {eval_results.get('eval_precision_macro', 0):.4f}")
print(f"    Macro Recall    : {eval_results.get('eval_recall_macro',    0):.4f}")
print(f"    Macro F1        : {eval_results.get('eval_f1_macro',        0):.4f}")
print(f"\n  Per-Class Breakdown:")
print(f"  {'Class':<{W}} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print(SEP)
 
for name in CLASS_NAMES:
    p = eval_results.get(f'eval_precision_{name}', 0)
    r = eval_results.get(f'eval_recall_{name}',    0)
    f = eval_results.get(f'eval_f1_{name}',        0)
    print(f"  {name:<{W}} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
 
print(SEP)
print()
 
# ============================================================
# Full sklearn classification report (for reference)
# ============================================================
print("  Full Classification Report:")
preds_all  = trainer.predict(test_dataset).predictions.argmax(-1)
print(classification_report(
    test_labels,
    preds_all,
    target_names=CLASS_NAMES,
    digits=4
))
 
# ============================================================
# STEP 9: Save Model
# ============================================================
print("Saving model...")
import os
os.makedirs("model", exist_ok=True)
 
torch.save(model.state_dict(), "model/model.pt")
print("✅ Model saved to ./model/model.pt")
 
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved to ./model/tokenizer.pkl")
 
label_mapping_info = {
    'classes'       : label_encoder.classes_.tolist(),
    'label_mapping' : label_mapping,
    'sorted_classes': CLASS_NAMES
}
with open("model/label_mapping.pkl", "wb") as f:
    pickle.dump(label_mapping_info, f)
print("✅ Label mapping saved to ./model/label_mapping.pkl")
print(f"\n  Saved mapping:")
for name, idx in sorted(label_mapping_info['label_mapping'].items()):
    print(f"    {idx} -> '{name}'")
 
print("\n✅ All files saved successfully!")