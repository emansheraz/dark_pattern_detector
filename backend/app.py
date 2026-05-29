from fastapi import FastAPI
from pydantic import BaseModel

import torch
import torch.nn.functional as F

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

# ============================================================
# APP
# ============================================================
app = FastAPI()

# ============================================================
# DEVICE
# ============================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = "../model"

tokenizer = BertTokenizer.from_pretrained(
    MODEL_PATH
)

model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH
)

class_names = torch.load(
    f"{MODEL_PATH}/class_names.pt"
)

model.to(device)

model.eval()

# ============================================================
# REQUEST FORMAT
# ============================================================
class TextRequest(BaseModel):
    text: str

# ============================================================
# HOME
# ============================================================
@app.get("/")
def home():

    return {
        "message": "Dark Pattern Detector API Running"
    }

# ============================================================
# PREDICT
# ============================================================
@app.post("/predict")
def predict(request: TextRequest):

    text = request.text

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

    probs = F.softmax(logits, dim=1)

    pred_id = torch.argmax(probs, dim=1).item()

    prediction = class_names[pred_id]

    confidence = probs[0][pred_id].item() * 100

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2)
    }