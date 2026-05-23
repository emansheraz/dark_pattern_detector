# 🕵️‍♂️ Dark Pattern Detector
### AI-Powered Detection of Deceptive E-Commerce Practices Using NLP and Transformer Models

Dark patterns are deceptive user interface and marketing techniques designed to manipulate users into making unintended decisions during online shopping experiences. These practices include fake urgency, hidden charges, disguised advertisements, and misleading promotional messages.

This project presents an **AI-powered Dark Pattern Detection System** that automatically identifies manipulative e-commerce text using advanced Natural Language Processing (NLP) and Transformer-based Deep Learning models. By leveraging Google's **BERT architecture**, the system understands contextual meaning in text rather than relying on basic keyword matching, making the detection process significantly more intelligent and accurate.

---

## 🚀 Key Features
* 🧠 **Transformer-Based NLP:** Fine-tuned `bert-base-uncased` language model for deep contextual analysis.
* 🏷️ **Multi-Class Classification:** Robust categorization of deceptive text into distinct dark pattern behaviors.
* ⚡ **Real-Time Prediction Pipeline:** Instant inference script to test UI copy on the fly.
* ⚖️ **Imbalance-Aware Training:** Utilizing stratified sampling and weighted evaluation metrics to handle skewed class distributions.
* 📈 **High Performance:** Attained an exceptional **95.27% accuracy** rate on unseen test data.

---

## 📊 Dark Pattern Categories & Dataset

The dataset consists of labeled e-commerce text samples mapped across four distinct classes, which have been thoroughly cleaned, preprocessed, and encoded for model training.

### Dataset Distribution
| Label | Description | Example | Samples |
| :--- | :--- | :--- | :--- |
| **Normal** | Non-manipulative, standard user interface text | *“Free delivery on orders above $50”* | 1,355 |
| **Fake Urgency** | Creates pressure through artificial scarcity or countdown tactics | *“Only 2 items left!”* | 874 |
| **Hidden Cost** | Reveals additional charges late in the user's checkout journey | *“Extra fees will apply at payment.”* | 398 |
| **Disguised Ad** | Advertisements presented natively as organic site content | *“Sponsored products you may like”* | 226 |

---

## 🧠 AI Architecture & Technical Workflow

Unlike traditional machine learning approaches that rely strictly on word frequency, BERT reads text bidirectionally to capture full sentence context.

```text
  User Input Text
        ↓
  BERT Tokenizer  → (Generates Attention Masks & Token IDs)
        ↓
 Fine-Tuned BERT  → (Transfer Learning via bert-base-uncased)
        ↓
  Linear Dropout  → (Classification Head)
        ↓
    Prediction    → [fake_urgency | hidden_cost | disguised_ad | normal]
## 🧠 AI Concepts Implemented
* **Transfer Learning & BERT Fine-Tuning:** Leveraging pre-trained language representations and optimizing them for task-specific sequence classification.
* **Subword Tokenization (`BertTokenizer`):** Breaking down text strings into efficient token inputs that capture vocabulary variations cleanly.
* **Stratified Train-Test Splitting:** Implementing an 80-20 split that strictly preserves the original dataset's percentage of samples for each class.
* **Weighted Optimization Metrics:** Utilizing loss-weighting mechanics to keep evaluation fair and accurate despite imbalanced class populations.

---

## ⚙️ Model Training & Performance

The model was built using the Hugging Face `transformers` ecosystem and optimized natively on `PyTorch`.

### Training Configuration
| Parameter | Value |
| :--- | :--- |
| **Base Model** | `bert-base-uncased` |
| **Framework / Library** | PyTorch & Hugging Face Transformers |
| **Epochs** | 3 |
| **Batch Size** | 8 |

### Evaluation Metrics
| Metric | Score |
| :--- | :--- |
| **Accuracy** | **95.27%** |
| **Precision** | **95.21%** |
| **Recall** | **95.27%** |
| **F1-Score** | **95.14%** |

> 📌 **Note:** Loss values decreased consistently throughout the 3 training epochs without signs of overfitting, demonstrating highly effective convergence.
## 📂 Project Structure
```text
dark_pattern_detector/
│
├── data/
│   └── dataset.csv          # Labeled e-commerce text data
│
├── model/                   # Local saved transformer assets
│   ├── config.json
│   ├── tokenizer.json
│   └── model.safetensors
│
├── results/                 # Training checkpoints (local only)
│
├── train.py                 # Preprocessing, training, and evaluation script
├── predict.py               # Inference script for testing custom text strings
├── requirements.txt         # Project software dependencies
└── README.md                # Project documentation
