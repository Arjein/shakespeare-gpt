import os
import torch
from tqdm import tqdm
from bigram_model import BLOCK_SIZE

BATCH_SIZE = 64
eval_iters = 50

def train_model(model, max_iters, learning_rate, train_data, val_data, device, path):
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
    model = model.to(device)
    for iters in tqdm(range(max_iters)):    
        if iters % eval_iters == 0:
            losses = estimate_loss(model, train_data, val_data, device)
            print(f'step {iters}: Train Loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}')
        
        xb, yb = get_batch('train',train_data, val_data, device)

        logits, loss = model(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    save_model(path, model)

# Data Loading
def get_batch(split, train_data, val_data, device):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i: i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1: i+BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x , y

@torch.inference_mode()
def estimate_loss(model, train_data, val_data, device):
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, y = get_batch(split,train_data, val_data, device=device)
            X, y = X.to(device), y.to(device)
            logits, loss = model(X, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    
    return out 



def save_model(PATH, model):
    os.makedirs('..models', exist_ok=True)
    torch.save(model.state_dict(), f'../models/{PATH}.pth')

