import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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
# Load the balanced dataset created by dataset.py
df = pd.read_csv("dark_patterns_dataset_clean.csv")

# Keep only needed columns
df = df[['text', 'label']]

# Drop empty rows
df = df.dropna()

print(f"✅ Dataset loaded: {len(df)} examples")
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution BEFORE encoding:")
print(df['label'].value_counts())
print()

# -----------------------------
# STEP 2: Encode Labels
# ============================
print("Encoding labels...")
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Create explicit label mapping (sklearn sorts alphabetically)
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(f"\n🔍 LABEL MAPPING (index -> class name):")
print(f"   Alphabetical order (how sklearn encodes):")
for idx, class_name in enumerate(sorted(label_encoder.classes_)):
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
    use_cpu=False if torch.cuda.is_available() else True  # ✅ Enable GPU if available
)

print(f"✅ Training will use: {device}\n")

# -----------------------------
# STEP 7: Evaluation Metrics
# -----------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

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

# Evaluate
print("\n" + "=" * 60)
print("EVALUATING ON TEST SET...")
print("=" * 60 + "\n")

eval_results = trainer.evaluate()

print("\nFinal Results:")
print(f"Accuracy:  {eval_results.get('eval_accuracy', 'N/A'):.4f}")
print(f"Precision: {eval_results.get('eval_precision', 'N/A'):.4f}")
print(f"Recall:    {eval_results.get('eval_recall', 'N/A'):.4f}")
print(f"F1 Score:  {eval_results.get('eval_f1', 'N/A'):.4f}\n")

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
    'sorted_classes': sorted(label_encoder.classes_)  # Explicit sorted order
}
with open("model/label_mapping.pkl", "wb") as f:
    pickle.dump(label_mapping_info, f)
print("✅ Label mapping saved to ./model/label_mapping.pkl")
print(f"\nSaved mapping info:")
print(f"  Classes: {label_mapping_info['classes']}")
print(f"  Sorted: {label_mapping_info['sorted_classes']}")
print(f"  Mapping: {label_mapping_info['label_mapping']}")

print("\n✅ All models saved successfully!")