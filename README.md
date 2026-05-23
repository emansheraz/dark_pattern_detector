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
AI Concepts ImplementedTransfer Learning & BERT Fine-TuningSubword Tokenization (BertTokenizer)Stratified Train-Test Splitting (80-20 Split)Weighted Optimization Metrics for Imbalanced Datasets⚙️ Model Training & PerformanceThe model was built using the Hugging Face transformers ecosystem and optimized natively on PyTorch.Training ConfigurationParameterValueBase Modelbert-base-uncasedFramework / LibraryPyTorch & Hugging Face TransformersEpochs3Batch Size8Evaluation MetricsMetricScoreAccuracy95.27%Precision95.21%Recall95.27%F1-Score95.14%📌 Note: Loss values decreased consistently throughout the 3 epochs without signs of overfitting, demonstrating effective convergence.📂 Project StructurePlaintextdark_pattern_detector/
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
🛠️ Installation & Usage1. Clone the RepositoryBashgit clone [https://github.com/EmanSheraz/dark_pattern_detector.git](https://github.com/EmanSheraz/dark_pattern_detector.git)
cd dark_pattern_detector
2. Install DependenciesBashpip install -r requirements.txt
3. Train the ModelTo run the full pipeline (data loading, preprocessing, tokenization, fine-tuning, and evaluation):Bashpython train.py
4. Run Real-Time PredictionsUse the prediction utility script to test custom UI strings against the model:Bashpython predict.py
Example Input: predict("Hurry! Only 2 items left!")Output: fake_urgency🔮 Future Improvements🧩 Browser Extension Integration: For real-time scanning of live website pages.👁️ OCR-Based Screenshot Detection: Analyzing visual UI layouts alongside text metadata.📊 Explainable AI (XAI): Visualizing specific token attention weights to explain why text was flagged.📈 Dashboard Analytics: Tracking user trends and logging dark pattern counts across popular sites.🛡️ Research & Ethical ImpactThis system contributes toward ethical AI deployment, consumer protection, and digital fairness. By putting automated defensive tools in the hands of everyday web users, this project aims to promote digital transparency and safeguard consumers from predatory online design patterns.Author: Eman SherazProject Type: AI & NLP-Based Final Year ProjectLicense: Educational and Research Use Only
