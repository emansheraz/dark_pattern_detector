from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model
model = BertForSequenceClassification.from_pretrained("model/")
tokenizer = BertTokenizer.from_pretrained("model/")

model.eval()  # IMPORTANT

# Label mapping
labels_map = {
    0: "disguised_ad",
    1: "fake_urgency",
    2: "hidden_cost",
    3: "normal"
}

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_id = torch.argmax(logits, dim=1).item()

    return labels_map[predicted_class_id]

# Test
text = "Hurry! Only 2 items left!"
print("Prediction:", predict("Extra charges will be added at checkout"))