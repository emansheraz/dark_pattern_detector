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
from torch.utils.data import Dataset
from lime.lime_text import LimeTextExplainer

# ============================================================
# LOAD MODEL + DATA  (already trained — just load)
# ============================================================
print("\nLoading saved model and tokenizer...")

MODEL_DIR  = "model/"
DATA_PATH  = "data/dataset.csv"

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model     = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load + encode dataset
df = pd.read_csv(DATA_PATH)[['text', 'label']].dropna().reset_index(drop=True)

label_encoder = LabelEncoder()
df['label']   = label_encoder.fit_transform(df['label'])
target_names  = list(label_encoder.classes_)
num_classes   = len(target_names)

# Same split as training so test set is identical
_, test_texts, _, test_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"Test set: {len(test_texts)} samples")
print(f"Classes : {target_names}")

# ============================================================
# BERT INFERENCE HELPER
# Returns logits, probabilities, and predicted class for a
# list of texts. Runs in batches to avoid OOM.
# ============================================================
def bert_predict(texts, batch_size=16):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)          # shape: (N, num_classes)

print("\nRunning BERT inference on test set...")
bert_probs  = bert_predict(test_texts)
bert_preds  = bert_probs.argmax(axis=1)
bert_labels = np.array(test_labels)

# ============================================================
# SECTION 1 — PER-CLASS ROC AUC CURVES
#
# Strategy: one-vs-rest binarisation
#   For each class C, treat it as positive (1) and all other
#   classes as negative (0). Plot the ROC curve for each.
#   AUC close to 1.0 = excellent separation for that class.
# ============================================================
print("\n── ROC AUC Curves ──────────────────────────────────────")

labels_bin = label_binarize(bert_labels, classes=list(range(num_classes)))

colors = ['#534AB7', '#0F6E56', '#D85A30', '#185FA5']

fig, axes = plt.subplots(1, num_classes, figsize=(18, 5))
fig.suptitle("Per-Class ROC Curves — BERT Dark Pattern Detector",
             fontsize=14, fontweight='bold', y=1.02)

roc_aucs = {}

for i, (class_name, color) in enumerate(zip(target_names, colors)):
    fpr, tpr, _ = roc_curve(labels_bin[:, i], bert_probs[:, i])
    roc_auc      = auc(fpr, tpr)
    roc_aucs[class_name] = roc_auc

    ax = axes[i]
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    ax.fill_between(fpr, tpr, alpha=0.08, color=color)

    ax.set_title(class_name, fontsize=12)
    ax.set_xlabel("False Positive Rate")
    if i == 0:
        ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("roc_auc_curves.png", dpi=150, bbox_inches='tight')
plt.show()

print("Per-class AUC scores:")
for cls, score in roc_aucs.items():
    print(f"  {cls:<18} AUC = {score:.4f}")

# ============================================================
# SECTION 2 — MISCLASSIFIED EXAMPLES WITH CONFIDENCE SCORES
#
# Shows exactly which texts the model got wrong, what it
# predicted, what the true label was, and how confident it
# was. Essential for error analysis in your viva.
# ============================================================
print("\n── Misclassified Examples ──────────────────────────────")

misclassified = []

for i, (true, pred) in enumerate(zip(bert_labels, bert_preds)):
    if true != pred:
        confidence    = bert_probs[i][pred] * 100
        true_conf     = bert_probs[i][true] * 100
        misclassified.append({
            'text':            test_texts[i][:120] + '...' if len(test_texts[i]) > 120 else test_texts[i],
            'true_label':      target_names[true],
            'predicted_label': target_names[pred],
            'confidence_%':    round(confidence, 1),
            'true_class_prob': round(true_conf, 1)
        })

miss_df = pd.DataFrame(misclassified)
miss_df = miss_df.sort_values('confidence_%', ascending=False)

print(f"\nTotal misclassified: {len(miss_df)} / {len(test_texts)}")
print(f"Error rate: {len(miss_df)/len(test_texts)*100:.2f}%\n")

# Print top 10 most confidently wrong predictions
print("Top 10 most confidently wrong predictions:")
print("─" * 80)
for _, row in miss_df.head(10).iterrows():
    print(f"  TEXT    : {row['text']}")
    print(f"  TRUE    : {row['true_label']:<18}  (model gave it {row['true_class_prob']}% confidence)")
    print(f"  PREDICTED: {row['predicted_label']:<18}  (model was {row['confidence_%']}% confident — WRONG)")
    print("─" * 80)

# Save full misclassification table
miss_df.to_csv("misclassified_examples.csv", index=False)
print(f"\nFull table saved → misclassified_examples.csv")

# Plot: confusion heatmap of only misclassified samples
if len(miss_df) > 0:
    miss_true = [target_names.index(x) for x in miss_df['true_label']]
    miss_pred = [target_names.index(x) for x in miss_df['predicted_label']]
    miss_cm   = confusion_matrix(miss_true, miss_pred,
                                  labels=list(range(num_classes)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Full confusion matrix
    cm_full = confusion_matrix(bert_labels, bert_preds)
    sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names, ax=ax1)
    ax1.set_title("Full Confusion Matrix")
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")

    # Confidence distribution of wrong predictions
    ax2.hist(miss_df['confidence_%'], bins=15, color='#D85A30', edgecolor='white')
    ax2.set_title("Confidence Distribution — Wrong Predictions")
    ax2.set_xlabel("Model Confidence (%)"); ax2.set_ylabel("Count")
    ax2.axvline(miss_df['confidence_%'].mean(), color='black',
                linestyle='--', label=f"Mean = {miss_df['confidence_%'].mean():.1f}%")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("misclassification_analysis.png", dpi=150)
    plt.show()

# ============================================================
# SECTION 3 — LIME EXPLAINABILITY
#
# LIME (Local Interpretable Model-agnostic Explanations)
# perturbs the input text (removes words one at a time) and
# watches how the prediction changes. Words that change the
# prediction most are highlighted as important.
#
# For BERT: we wrap predict_proba so LIME can call it.
# ============================================================
print("\n── LIME Explainability ─────────────────────────────────")

def bert_predict_proba(texts):
    """LIME-compatible wrapper — returns probability array."""
    return bert_predict(list(texts))

explainer = LimeTextExplainer(
    class_names=target_names,
    random_state=42
)

# Pick 1 example per class to explain
print("\nGenerating LIME explanations (1 per class)...")

lime_samples = {}
for class_id, class_name in enumerate(target_names):
    # Find a correctly predicted example for this class
    for i, (true, pred) in enumerate(zip(bert_labels, bert_preds)):
        if true == class_id and pred == class_id:
            lime_samples[class_name] = (i, test_texts[i])
            break

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()
fig.suptitle("LIME Explanations — Words That Triggered Each Dark Pattern",
             fontsize=14, fontweight='bold')

for idx, (class_name, (sample_idx, sample_text)) in enumerate(lime_samples.items()):
    print(f"  Explaining: [{class_name}] → \"{sample_text[:80]}...\"")

    exp = explainer.explain_instance(
        sample_text,
        bert_predict_proba,
        num_features=10,    # top 10 most important words
        num_samples=100,    # perturbations (higher = more accurate but slower)
        labels=[label_encoder.transform([class_name])[0]]
    )

    label_id   = label_encoder.transform([class_name])[0]
    word_weights = exp.as_list(label=label_id)

    # Sort by absolute importance
    word_weights.sort(key=lambda x: abs(x[1]), reverse=True)
    words  = [w[0] for w in word_weights]
    scores = [w[1] for w in word_weights]
    colors_bar = ['#0F6E56' if s > 0 else '#D85A30' for s in scores]

    ax = axes[idx]
    bars = ax.barh(words[::-1], scores[::-1], color=colors_bar[::-1], edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f"Class: {class_name}\n\"{sample_text[:60]}...\"",
                 fontsize=10)
    ax.set_xlabel("LIME Importance Score")
    ax.grid(True, alpha=0.3, axis='x')

    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#0F6E56', label='Supports prediction'),
        Patch(facecolor='#D85A30', label='Against prediction')
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

plt.tight_layout()
plt.savefig("lime_explanations.png", dpi=150, bbox_inches='tight')
plt.show()
print("LIME plots saved → lime_explanations.png")

# ============================================================
# SECTION 4 — BERT vs BASELINE COMPARISON
#
# We train two traditional ML baselines:
#   1. TF-IDF + LinearSVC    (classic strong baseline)
#   2. TF-IDF + Logistic Regression  (probabilistic baseline)
#
# Then compare accuracy, macro F1, and per-class F1 against
# your BERT model. This justifies your architecture choice.
# ============================================================
print("\n── Baseline Comparison ─────────────────────────────────")

train_texts_raw, _, train_labels_raw, _ = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# ── Baseline 1: TF-IDF + LinearSVC ──────────────────────
print("\nTraining Baseline 1: TF-IDF + LinearSVC...")

svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),   # unigrams + bigrams
        sublinear_tf=True
    )),
    # CalibratedClassifierCV wraps LinearSVC to give probabilities
    # needed for ROC AUC — LinearSVC alone has no predict_proba
    ('clf', CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=3))
])

svm_pipeline.fit(train_texts_raw, train_labels_raw)
svm_preds  = svm_pipeline.predict(test_texts)
svm_probs  = svm_pipeline.predict_proba(test_texts)

svm_acc = accuracy_score(test_labels, svm_preds)
_, _, svm_macro_f1, _ = precision_recall_fscore_support(
    test_labels, svm_preds, average='macro', zero_division=0)

print(f"  TF-IDF + SVM  →  Accuracy: {svm_acc:.4f}  |  Macro F1: {svm_macro_f1:.4f}")

# ── Baseline 2: TF-IDF + Logistic Regression ────────────
print("Training Baseline 2: TF-IDF + Logistic Regression...")

lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ('clf', LogisticRegression(max_iter=1000, C=1.0))
])

lr_pipeline.fit(train_texts_raw, train_labels_raw)
lr_preds = lr_pipeline.predict(test_texts)
lr_probs = lr_pipeline.predict_proba(test_texts)

lr_acc = accuracy_score(test_labels, lr_preds)
_, _, lr_macro_f1, _ = precision_recall_fscore_support(
    test_labels, lr_preds, average='macro', zero_division=0)

print(f"  TF-IDF + LR   →  Accuracy: {lr_acc:.4f}  |  Macro F1: {lr_macro_f1:.4f}")

# ── BERT scores ──────────────────────────────────────────
bert_acc = accuracy_score(bert_labels, bert_preds)
_, _, bert_macro_f1, _ = precision_recall_fscore_support(
    bert_labels, bert_preds, average='macro', zero_division=0)

print(f"  BERT (ours)   →  Accuracy: {bert_acc:.4f}  |  Macro F1: {bert_macro_f1:.4f}")

# ── Per-class F1 for all three models ───────────────────
_, _, svm_f1_per,  _ = precision_recall_fscore_support(test_labels, svm_preds,  average=None, zero_division=0)
_, _, lr_f1_per,   _ = precision_recall_fscore_support(test_labels, lr_preds,   average=None, zero_division=0)
_, _, bert_f1_per, _ = precision_recall_fscore_support(bert_labels, bert_preds, average=None, zero_division=0)

# ── Summary table ────────────────────────────────────────
summary = pd.DataFrame({
    'Model':     ['TF-IDF + SVM', 'TF-IDF + LR', 'BERT (ours)'],
    'Accuracy':  [svm_acc,        lr_acc,         bert_acc],
    'Macro F1':  [svm_macro_f1,   lr_macro_f1,    bert_macro_f1],
})
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)
print(summary.to_string(index=False))
print("="*50)

# ── Visualise comparison ─────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

model_labels = ['TF-IDF\n+ SVM', 'TF-IDF\n+ LR', 'BERT\n(ours)']
bar_colors   = ['#888780', '#B4B2A9', '#534AB7']   # gray, gray, purple for BERT

# Plot 1 — Accuracy
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(model_labels,
               [svm_acc, lr_acc, bert_acc],
               color=bar_colors, edgecolor='white', width=0.5)
ax1.set_ylim(0.5, 1.0)
ax1.set_title("Overall Accuracy", fontweight='bold')
ax1.set_ylabel("Accuracy")
for bar, val in zip(bars, [svm_acc, lr_acc, bert_acc]):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005,
             f"{val*100:.1f}%", ha='center', va='bottom', fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2 — Macro F1
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(model_labels,
               [svm_macro_f1, lr_macro_f1, bert_macro_f1],
               color=bar_colors, edgecolor='white', width=0.5)
ax2.set_ylim(0.5, 1.0)
ax2.set_title("Macro F1 Score", fontweight='bold')
ax2.set_ylabel("Macro F1")
for bar, val in zip(bars, [svm_macro_f1, lr_macro_f1, bert_macro_f1]):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005,
             f"{val:.3f}", ha='center', va='bottom', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3 — Per-class F1 grouped bar
ax3 = fig.add_subplot(gs[1, :])
x      = np.arange(num_classes)
width  = 0.25

ax3.bar(x - width, svm_f1_per,  width, label='TF-IDF + SVM', color='#888780', edgecolor='white')
ax3.bar(x,         lr_f1_per,   width, label='TF-IDF + LR',  color='#B4B2A9', edgecolor='white')
ax3.bar(x + width, bert_f1_per, width, label='BERT (ours)',   color='#534AB7', edgecolor='white')

ax3.set_xticks(x)
ax3.set_xticklabels(target_names, fontsize=11)
ax3.set_ylim(0, 1.1)
ax3.set_title("Per-Class F1 Score — All Models", fontweight='bold')
ax3.set_ylabel("F1 Score")
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Annotate BERT bars only
for i, val in enumerate(bert_f1_per):
    ax3.text(i + width, val + 0.02, f"{val:.2f}",
             ha='center', va='bottom', fontsize=9, color='#534AB7', fontweight='bold')

plt.suptitle("BERT vs Baseline Models — Dark Pattern Detection",
             fontsize=14, fontweight='bold', y=1.01)

plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# ── ROC comparison on disguised_ad (weakest class) ───────
fig, ax = plt.subplots(figsize=(7, 6))

class_idx = target_names.index('disguised_ad')
lab_bin   = (bert_labels == class_idx).astype(int)

for probs, name, color in [
    (svm_probs,  'TF-IDF + SVM', '#888780'),
    (lr_probs,   'TF-IDF + LR',  '#B4B2A9'),
    (bert_probs, 'BERT (ours)',   '#534AB7'),
]:
    fpr, tpr, _ = roc_curve(lab_bin, probs[:, class_idx])
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{name}  AUC={roc_auc:.3f}")

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
ax.set_title("ROC Curve — disguised_ad class (hardest class)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_comparison_disguised_ad.png", dpi=150)
plt.show()

print("\n" + "="*60)
print("ALL EVALUATION COMPLETE")
print("="*60)
print(f"  Accuracy  — BERT: {bert_acc*100:.2f}%  |  SVM: {svm_acc*100:.2f}%  |  LR: {lr_acc*100:.2f}%")
print(f"  Macro F1  — BERT: {bert_macro_f1:.4f}  |  SVM: {svm_macro_f1:.4f}  |  LR: {lr_macro_f1:.4f}")
print(f"\nFiles saved:")
print("  roc_auc_curves.png")
print("  misclassification_analysis.png")
print("  misclassified_examples.csv")
print("  lime_explanations.png")
print("  model_comparison.png")
print("  roc_comparison_disguised_ad.png")