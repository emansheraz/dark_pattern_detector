import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import pickle

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
print("Loading model...")

# Load model state dict from .pt file
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
model.load_state_dict(torch.load("model/model.pt", map_location="cpu"))

# Load tokenizer from .pkl file
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label mapping from .pkl file
with open("model/label_mapping.pkl", "rb") as f:
    label_mapping_info = pickle.load(f)

# Create explicit mapping: index -> label (using sorted order)
sorted_classes = label_mapping_info['sorted_classes']
labels_map = {idx: class_name for idx, class_name in enumerate(sorted_classes)}

# Evaluation mode
model.eval()

# Descriptions for each category
descriptions = {
    "disguised_ad": "📢 This text looks like an ad disguised as organic content",
    "fake_urgency": "⚠️ This text creates artificial urgency or scarcity",
    "hidden_cost": "💰 This text hides charges that appear later",
    "normal": "✅ This text is normal and transparent"
}

print(f"✅ Model loaded successfully!")
print(f"\n🔍 LOADED LABEL MAPPING (index -> class name):")
for idx, label in labels_map.items():
    print(f"   {idx} -> '{label}'")
print(f"\nFull mapping from file: {label_mapping_info['label_mapping']}\n")

def predict(text):
    """
    Predict dark pattern category and return results with confidence
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
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
    
    # Get probabilities using softmax
    probabilities = F.softmax(logits, dim=1)
    
    # Get predicted class
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_label = labels_map[predicted_class_id]
    
    # Get confidence score
    confidence = probabilities[0][predicted_class_id].item() * 100
    
    # Get all probabilities
    all_probs = {}
    for class_id, label in labels_map.items():
        prob = probabilities[0][class_id].item() * 100
        all_probs[label] = prob
    
    return {
        'prediction': predicted_label,
        'confidence': confidence,
        'all_probabilities': all_probs
    }

def print_prediction_result(text, result):
    """
    Pretty print prediction results for easy interpretation
    """
    prediction = result['prediction']
    confidence = result['confidence']
    all_probs = result['all_probabilities']
    description = descriptions[prediction]
    
    # Print the input text
    print("\n" + "=" * 70)
    print("📝 INPUT TEXT:")
    print("-" * 70)
    print(f'"{text}"')
    print("=" * 70)
    
    # Print main prediction
    print("\n🎯 PREDICTION RESULT:")
    print("-" * 70)
    print(f"Category: {prediction.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\n{description}")
    print("=" * 70)
    
    # Print all probabilities
    print("\n📊 DETAILED BREAKDOWN (All Categories):")
    print("-" * 70)
    
    # Sort by probability (highest first)
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    
    for i, (label, prob) in enumerate(sorted_probs):
        # Add emoji based on position
        if i == 0:
            emoji = "🥇"  # Gold medal for top prediction
        elif i == 1:
            emoji = "🥈"  # Silver
        elif i == 2:
            emoji = "🥉"  # Bronze
        else:
            emoji = "  "  # No medal
        
        # Create a visual bar
        bar_length = int(prob / 5)  # Scale to 0-20
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        print(f"{emoji} {label:15s} {bar} {prob:6.2f}%")
    
    print("=" * 70 + "\n")

# Test with sample texts
print("\n" + "🚀" * 35)
print("DARK PATTERN DETECTOR - PREDICTION DEMO")
print("🚀" * 35 + "\n")

# Sample test cases
test_cases = [
    "Extra charges will be added at checkout",
    "Only 2 items left in stock!",
    "Free shipping on orders over $50",
    "Sponsored products you may like",
    "Hurry! This offer expires in 1 hour!"
]

print("Running predictions on sample texts...\n")

for text in test_cases:
    result = predict(text)
    print_prediction_result(text, result)

# Interactive mode
print("\n" + "=" * 70)
print("INTERACTIVE MODE")
print("=" * 70)
print("Enter your own text to classify (or type 'quit' to exit)\n")

while True:
    user_text = input("📝 Enter text: ").strip()
    
    if user_text.lower() == 'quit':
        print("\n✅ Thank you for using Dark Pattern Detector!")
        break
    
    if not user_text:
        print("⚠️ Please enter some text!\n")
        continue
    
    result = predict(user_text)
    print_prediction_result(user_text, result)