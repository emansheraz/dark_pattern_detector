import torch
from transformers import BertTokenizer, BertForSequenceClassification

# -----------------------------
# STEP 1: Device Configuration
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device: {device}")

# -----------------------------
# STEP 2: Load Model & Tokenizer
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    "model/"
).to(device)

tokenizer = BertTokenizer.from_pretrained("model/")

# Evaluation mode
model.eval()

# -----------------------------
# STEP 3: Label Mapping
# -----------------------------
labels_map = {
    0: "disguised_ad",
    1: "fake_urgency",
    2: "hidden_cost",
    3: "normal"
}

# -----------------------------
# STEP 4: Prediction Function
# -----------------------------
def predict(text):

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    # Move tensors to GPU/CPU
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Disable gradients
    with torch.no_grad():

        outputs = model(**inputs)

        logits = outputs.logits

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=1)

        # Get prediction
        predicted_class_id = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = torch.max(
            probabilities
        ).item()

    return {
        "prediction": labels_map[predicted_class_id],
        "confidence": round(confidence * 100, 2)
    }

# -----------------------------
# STEP 5: Test Predictions
# -----------------------------
if __name__ == "__main__":

    examples = [
        "Hurry! Only 2 items left in stock!",
        "Extra delivery charges will be added at checkout.",
        "Sponsored products recommended for you.",
        "Free shipping on orders above $50."
    ]

    print("\nDark Pattern Predictions:\n")

    for text in examples:

        result = predict(text)

        print(f"Text: {text}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}%")

        print("-" * 60)