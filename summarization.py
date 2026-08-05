# models/summarization.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "facebook/bart-large-cnn"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def summarize_text(text):

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )


    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )


    # Decode output
    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )


    return summary



# Test
if __name__ == "__main__":

    text = """
    Artificial intelligence is transforming industries around the world.
    Companies are using machine learning, deep learning, and natural language
    processing to automate tasks and improve decision making.
    """

    result = summarize_text(text)

    print("Summary:")
    print(result)