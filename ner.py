# models/ner.py

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch


MODEL_NAME = "dslim/bert-base-NER"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)


def extract_entities(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(
        outputs.logits,
        dim=2
    )


    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )


    entities = []

    for token, prediction in zip(tokens, predictions[0]):

        label = model.config.id2label[prediction.item()]

        if label != "O":
            entities.append(
                {
                    "word": token,
                    "entity": label
                }
            )

    return entities



# Test
if __name__ == "__main__":

    text = "Elon Musk founded SpaceX in California."

    result = extract_entities(text)

    for item in result:
        print(item)