# models/question_answering.py

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


MODEL_NAME = "deepset/roberta-base-squad2"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)


def answer_question(context, question):

    # Tokenize context + question
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True
    )


    # Prediction
    with torch.no_grad():
        outputs = model(**inputs)


    # Start and end positions
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits)


    # Convert tokens to answer
    answer_tokens = inputs.input_ids[0][
        start_index:end_index + 1
    ]

    answer = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True
    )


    return answer



# Test
if __name__ == "__main__":

    context = """
    Pakistan is a country in South Asia.
    Islamabad is the capital city of Pakistan.
    """

    question = "What is the capital of Pakistan?"

    result = answer_question(
        context,
        question
    )

    print("Answer:", result)