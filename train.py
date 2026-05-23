import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
df = pd.read_csv("data/dataset.csv")

# Keep only needed columns
df = df[['text', 'label']]

# Drop empty rows
df = df.dropna()

# -----------------------------
# STEP 2: Encode Labels
# -----------------------------
print(df['label'].value_counts())
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

print("Label Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# -----------------------------
# STEP 3: Train-Test Split
# -----------------------------
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label'] 
)


# -----------------------------
# STEP 4: Load Tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

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
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4
)

# -----------------------------
# STEP 6: Training Arguments
# -----------------------------

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",  # Optional: evaluates after every epoch
    logging_dir="./logs",
    save_strategy="epoch",
    report_to="none"        # Prevents errors if you don't have WandB installed
)

# -----------------------------
# STEP 7: Evaluation Metrics
# -----------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
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
trainer.train()

# Evaluate
trainer.evaluate()

# -----------------------------
# STEP 9: Save Model
# -----------------------------
model.save_pretrained("model/")
tokenizer.save_pretrained("model/")

print("Model saved successfully!")