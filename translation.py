from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-ur"

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)


def translate_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True
    )

    translated = model.generate(**inputs)

    output = tokenizer.decode(
        translated[0],
        skip_special_tokens=True
    )

    return output