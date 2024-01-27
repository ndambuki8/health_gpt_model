import torch
import torch.nn as nn
import pandas as pd
from torch.nn import functional as F
from healthGptVersion import GPTLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the GPTLanguageModel class and load the saved model as shown in the previous response.

# Load the saved model
model = GPTLanguageModel()
model.load_state_dict(torch.load('gpt_model_final.pth'))
model = model.to(device)
model.eval()

# Function to generate a response
def generate_response(input_text, max_tokens=100):
     
    chars = sorted(list(set(input_text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
#     decode = lambda l: ''.join([itos[i] for i in l])
    decode = lambda l: ''.join([itos.get(i, '') for i in l])
          
    input_tokens = encode(input_text)
    context = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated_response = decode(model.generate(context, max_tokens)[0].tolist())
    return generated_response

# User interaction
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = generate_response(user_input)
    print("Model: " + response)
