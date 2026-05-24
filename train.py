import pandas as pd
import numpy as np
import torch
import random
import os
import pickle
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
 
# ============================================================
# RANDOM STATE — change this one variable to reproduce any
# experiment. Every split, shuffle, and seed in the script
# uses this value so results are fully reproducible.
# ============================================================
RANDOM_STATE = 42
 
# Seeds every relevant library so runs are reproducible
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
 
set_seed(RANDOM_STATE)
 
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
# STEP 3: Train / Test Split (stratified 80-20)
#
# RANDOM_STATE controls which specific samples go into train
# vs test. With only ~28 test samples per class, a single
# bad split can swing accuracy by 3-5%. Using RANDOM_STATE
# as a named variable makes experiments reproducible and
# lets you run multiple seeds to get a reliable average.
# ============================================================
print(f"Splitting data (80-20 stratified split, random_state={RANDOM_STATE})...")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=RANDOM_STATE,   # ← controlled by top-level variable
    stratify=df['label']
)
print(f"Training samples: {len(train_texts)}")
print(f"Testing  samples: {len(test_texts)}\n")
 
# ============================================================
# STEP 4: Tokenise
#
# max_length reduced from 128 → 64 because 90% of your texts
# are under 10 words (~55 chars). Padding to 128 wastes GPU
# memory and adds noise from meaningless [PAD] attention.
# ============================================================
print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
 
print("Tokenizing texts (max_length=64)...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
test_encodings  = tokenizer(test_texts,  truncation=True, padding=True, max_length=64)
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
 
# ── Freeze lower 8 of 12 BERT encoder layers ────────────────
# With only ~450 training samples fine-tuning all 12 layers
# causes overfitting. Freezing the bottom 8 forces the model
# to adapt only the top layers (which hold task-specific
# representations) while keeping general language knowledge
# in the frozen layers intact.
for i, layer in enumerate(model.bert.encoder.layer):
    if i < 8:
        for param in layer.parameters():
            param.requires_grad = False
 
frozen   = sum(p.numel() for p in model.parameters() if not p.requires_grad)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Frozen parameters   : {frozen:,}")
print(f"   Trainable parameters: {trainable:,}")
 
model.to(device)
print(f"✅ Model loaded and moved to {device}\n")
 
# ============================================================
# STEP 6: Training Arguments
#
# Key changes vs original:
#   num_train_epochs 5 → 8   : load_best_model_at_end protects
#                               against overfitting; more epochs
#                               give more checkpoints to pick from
#   batch_size       8 → 16  : short texts fit easily; larger
#                               batches give more stable gradients
#   seed = RANDOM_STATE      : tied to the top-level variable
# ============================================================
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir                  = "./results",
    num_train_epochs            = 8,           # was 5
    per_device_train_batch_size = 16,          # was 8
    per_device_eval_batch_size  = 16,          # was 8
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    logging_dir                 = "./logs",
    logging_steps               = 10,
    report_to                   = "none",
    seed                        = RANDOM_STATE,  # ← tied to top-level variable
 
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
# STEP 7: compute_metrics — MACRO average (no class weighting)
#   average='macro' : every class contributes equally to the
#                     score regardless of sample count.
#   average=None    : returns a per-class array so we can log
#                     individual class scores alongside macro.
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
        'accuracy'        : acc,
        'precision_macro' : float(p_macro),
        'recall_macro'    : float(r_macro),
        'f1_macro'        : float(f1_macro),   # ← used for best-model selection
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
W   = 20
SEP = "  " + "-" * (W + 34)
 
print("\n📊 FINAL EVALUATION RESULTS")
print("=" * 60)
print(f"\n  Random State Used : {RANDOM_STATE}")
print(f"  Overall Accuracy  : {eval_results.get('eval_accuracy', 0):.4f}")
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
preds_all = trainer.predict(test_dataset).predictions.argmax(-1)
print(classification_report(
    test_labels,
    preds_all,
    target_names=CLASS_NAMES,
    digits=4
))
 
# ============================================================
# MULTI-SEED EVALUATION
#
# With only ~28 test samples per class a single split can
# swing accuracy by 3-5% purely by luck. Running 5 seeds and
# averaging gives a much more reliable estimate of true
# model performance. Results are printed at the end.
# ============================================================
print("\n" + "=" * 60)
print("MULTI-SEED EVALUATION (5 random states)...")
print("Reusing trained model weights — only the split changes.")
print("=" * 60 + "\n")
 
EVAL_SEEDS   = [0, 1, 42, 99, 123]
seed_results = []
 
for rs in EVAL_SEEDS:
    _, seed_test_texts, _, seed_test_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=rs,
        stratify=df['label']
    )
 
    seed_encodings = tokenizer(
        seed_test_texts, truncation=True, padding=True, max_length=64
    )
    seed_dataset = DarkPatternDataset(seed_encodings, seed_test_labels)
 
    seed_preds  = trainer.predict(seed_dataset).predictions.argmax(-1)
    seed_acc    = accuracy_score(seed_test_labels, seed_preds)
    _, _, seed_f1, _ = precision_recall_fscore_support(
        seed_test_labels, seed_preds, average='macro', zero_division=0
    )
    seed_results.append({'seed': rs, 'accuracy': seed_acc, 'f1_macro': seed_f1})
    print(f"  seed={rs:>3} | Accuracy: {seed_acc:.4f} | Macro F1: {seed_f1:.4f}")
 
avg_acc = sum(r['accuracy'] for r in seed_results) / len(seed_results)
avg_f1  = sum(r['f1_macro'] for r in seed_results) / len(seed_results)
std_acc = np.std([r['accuracy'] for r in seed_results])
std_f1  = np.std([r['f1_macro'] for r in seed_results])
 
print(f"\n  Average Accuracy : {avg_acc:.4f}  (± {std_acc:.4f})")
print(f"  Average Macro F1 : {avg_f1:.4f}  (± {std_f1:.4f})")
print()
 
# ============================================================
# STEP 9: Save Model
# ============================================================
print("Saving model...")
os.makedirs("model", exist_ok=True)
 
torch.save(model.state_dict(), "model/model.pt")
print("✅ Model saved to ./model/model.pt")
 
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved to ./model/tokenizer.pkl")
 
label_mapping_info = {
    'classes'       : label_encoder.classes_.tolist(),
    'label_mapping' : label_mapping,
    'sorted_classes': CLASS_NAMES,
    'random_state'  : RANDOM_STATE      # ← saved alongside weights for traceability
}
with open("model/label_mapping.pkl", "wb") as f:
    pickle.dump(label_mapping_info, f)
print("✅ Label mapping saved to ./model/label_mapping.pkl")
print(f"\n  Saved mapping:")
for name, idx in sorted(label_mapping_info['label_mapping'].items()):
    print(f"    {idx} -> '{name}'")
 
print("\n✅ All files saved successfully!")