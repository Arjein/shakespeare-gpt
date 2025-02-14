import torch 
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
# Define the Hyperparameters

BLOCK_SIZE = 256
device = 'mps' if torch.mps.is_available() else 'cpu'
n_embd = 384
dropout = 0.2

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size, n_layer, n_head):
    super().__init__()

    self.token_embedding_table = nn.Embedding(vocab_size, n_embd, device=device)
    self.position_embedding_table = nn.Embedding(BLOCK_SIZE , n_embd, device=device)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd, device=device)
    self.lm_head = nn.Linear(n_embd, vocab_size, device=device)
  
  def forward(self, idx, targets= None):
    
    B, T = idx.shape

    tok_emb = self.token_embedding_table(idx) # (B, T, C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)

    x = tok_emb + pos_emb
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) # (B, T, vocab_size)

    if targets == None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B* T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
      
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]

        logits, loss = self(idx_cond)
        logits = logits[:, -1, :] 
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd, device=device)
        self.ln2 = nn.LayerNorm(n_embd, device=device)
    
    def forward(self, x):
       x = x+ self.sa(self.ln1(x))
       x = x+ self.ffwd(self.ln2(x))
       return x

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
      
      out = torch.cat([h(x) for h in self.heads], dim=-1)
      out = self.dropout(self.proj(out))
      return out

    
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
       B, T, C = x.shape
       k = self.key(x)
       q = self.query(x)
       v = self.value(x)

       wei = q @ k.transpose(-2, -1) * C ** -0.5
       wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
       wei = F.softmax(wei, dim=-1)
       wei = self.dropout(wei)

       v = self.value(x)
       out = wei @ v
       return out