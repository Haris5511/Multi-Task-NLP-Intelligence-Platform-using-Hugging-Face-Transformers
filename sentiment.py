# models/sentiment.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model Name
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Label Mapping
labels = {
    0: "Negative",
    1: "Positive"
}


def predict_sentiment(text):
    """
    Predict sentiment of input text.

    Args:
        text (str): Input sentence

    Returns:
        dict: {
            "label": "...",
            "confidence": 99.45
        }
    """

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    # Disable gradient calculation
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probabilities = torch.softmax(outputs.logits, dim=1)

    confidence, predicted_class = torch.max(probabilities, dim=1)

    return {
        "label": labels[predicted_class.item()],
        "confidence": round(confidence.item() * 100, 2)
    }


# Test
if __name__ == "__main__":

    text = input("Enter Text: ")

    result = predict_sentiment(text)

    print("\nPrediction")
    print("--------------------")
    print("Label      :", result["label"])
    print("Confidence :", result["confidence"], "%")