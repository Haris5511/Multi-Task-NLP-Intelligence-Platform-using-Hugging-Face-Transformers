from transformers import pipeline

# Load model once
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

def classify_text(text, labels):
    """
    text: Input sentence
    labels: List of labels
    """
    result = classifier(text, labels)

    return {
        "labels": result["labels"],
        "scores": result["scores"]
    }