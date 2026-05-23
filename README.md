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
