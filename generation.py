

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


MODEL_NAME = "gpt2"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def generate_text(prompt):

    # Tokenization
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )


    # Generate text
    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )


    # Decode result
    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


    return result



# Test
if __name__ == "__main__":

    prompt = "Artificial Intelligence is"

    result = generate_text(prompt)

    print(result)