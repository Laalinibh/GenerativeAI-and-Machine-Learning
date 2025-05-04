import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class FewShotLearner:
    def __init__(self, model_name='gpt2'):
        # Initialize GPT2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()  # We are in inference mode (no gradient updates)

    def encode_prompt(self, prompt):
        # Encode the prompt with a few-shot example
        return self.tokenizer.encode(prompt, return_tensors='pt')

    def generate_response(self, prompt, max_length=50):
        # Generate text from a given prompt
        input_ids = self.encode_prompt(prompt)
        
        # Generate predictions
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=max_length, num_return_sequences=1)
        
        # Decode the generated text
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

# Example usage of Few-Shot Learner for text generation
few_shot_model = FewShotLearner()

# Let's define a task and provide a few examples
prompt = """
Q: What is the capital of France?
A: The capital of France is Paris.

Q: What is 2 + 2?
A: 2 + 2 equals 4.

Q: Who wrote 'Hamlet'?
A: William Shakespeare wrote 'Hamlet'.

Q: What is the tallest mountain in the world?
A: The tallest mountain in the world is Mount Everest.
"""

# Generate an answer to a new question with the few-shot examples
response = few_shot_model.generate_response(prompt + "\nQ: What is the fastest animal on land?")
print(response)
