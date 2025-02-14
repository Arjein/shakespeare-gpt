import argparse
import torch
import os
from bigram_model import BigramLanguageModel
from train import train_model

torch.manual_seed(42)
device = 'mps' if torch.mps.is_available() else 'cpu'

# Define Argument Parser
parser = argparse.ArgumentParser(description="Bigram Language Model Training & Generation")
parser.add_argument("--mode", type=str, required=True, choices=["train", "generate"], help="Mode: train or generate")
parser.add_argument("--model_path", type=str, help="Path to the model file")
parser.add_argument("--max_iters", type=int, default=5000, help="Number of training iterations")


args = parser.parse_args()

# Load Data
with open("../input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long, device=device)
train_cutoff = int(data.shape[0] * 0.9)
train_data = data[:train_cutoff]
val_data = data[train_cutoff:]

learning_rate = 3e-4

# Train Function
def train(model_path, max_iters, learning_rate, device=device):
    model = BigramLanguageModel(vocab_size=vocab_size, n_head=6, n_layer=6)
    if os.path.exists(model_path):
        model_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(model_dict)
        model = model.to(device)
        print("Loaded existing model.")
        train_model(model=model, max_iters=max_iters, learning_rate=learning_rate, train_data=train_data, val_data=val_data, path=model_path)
    else:
        print("Training new model...")
        train_model(model=model, max_iters=max_iters, learning_rate=learning_rate, device=device, train_data=train_data, val_data=val_data, path=args.model_path)

# Generate Function
def generate(model_path, device=device):
    if not os.path.exists(model_path):
        print("Model not found. Train it first using '--mode train'.")
        return

    model_dict = torch.load(model_path, map_location=device)
    model = BigramLanguageModel(vocab_size=vocab_size, n_head=6, n_layer=6)
    model.load_state_dict(model_dict)
    model = model.to(device)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

# Execute Based on User Choice
if args.mode == "train":
    train(args.model_path, args.max_iters, learning_rate)
elif args.mode == "generate":
    generate(f'../models/{args.model_path}.pth')