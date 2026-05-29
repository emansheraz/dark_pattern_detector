import torch
import torch.nn.functional as F

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

# ============================================================
# DEVICE
# ============================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DARK PATTERN DETECTOR")
print("=" * 60)

print(f"Using device: {device}")

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = "./model"

print("\nLoading tokenizer...")

tokenizer = BertTokenizer.from_pretrained(
    MODEL_PATH
)

print("Loading model...")

model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH
)

class_names = torch.load(
    f"{MODEL_PATH}/class_names.pt"
)

model.to(device)

model.eval()

print("✅ Model loaded successfully!")

# ============================================================
# DESCRIPTIONS
# ============================================================
descriptions = {
    "disguised_ad": "📢 Advertisement disguised as normal content",
    "fake_urgency": "⚠️ Creates fake urgency or pressure",
    "hidden_cost": "💰 Contains hidden charges or extra fees",
    "normal": "✅ Normal transparent content"
}

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        logits = outputs.logits

    probabilities = F.softmax(logits, dim=1)

    predicted_class_id = torch.argmax(
        probabilities,
        dim=1
    ).item()

    predicted_label = class_names[predicted_class_id]

    confidence = (
        probabilities[0][predicted_class_id].item() * 100
    )

    all_probabilities = {}

    for i, label in enumerate(class_names):

        prob = probabilities[0][i].item() * 100

        all_probabilities[label] = round(prob, 2)

    return {
        "prediction": predicted_label,
        "confidence": round(confidence, 2),
        "all_probabilities": all_probabilities
    }

# ============================================================
# PRINT FUNCTION
# ============================================================
def print_prediction(text, result):

    prediction = result["prediction"]

    confidence = result["confidence"]

    all_probs = result["all_probabilities"]

    print("\n" + "=" * 70)

    print("📝 INPUT TEXT:")
    print("-" * 70)

    print(f'"{text}"')

    print("\n🎯 PREDICTION RESULT:")
    print("-" * 70)

    print(f"Category   : {prediction}")

    print(f"Confidence : {confidence:.2f}%")

    print(f"\n{descriptions[prediction]}")

    print("\n📊 ALL CLASS PROBABILITIES:")
    print("-" * 70)

    sorted_probs = sorted(
        all_probs.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for label, prob in sorted_probs:

        bar_length = int(prob / 5)

        bar = (
            "█" * bar_length +
            "░" * (20 - bar_length)
        )

        print(
            f"{label:15s} {bar} {prob:6.2f}%"
        )

    print("=" * 70)

# ============================================================
# INTERACTIVE MODE
# ============================================================
print("\n" + "=" * 60)
print("INTERACTIVE MODE")
print("Type 'quit' to exit")
print("=" * 60)

while True:

    text = input("\n📝 Enter text: ").strip()

    if text.lower() == "quit":

        print("\n✅ Exiting...")

        break

    if not text:

        print("⚠️ Empty input!")

        continue

    result = predict(text)

    print_prediction(text, result)