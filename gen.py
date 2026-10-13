import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import tiktoken
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_FILE = os.path.join(HERE, "checkpoints", "model.pt")

VOCAB_SIZE = 50257
BLOCK_SIZE = 1024
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 40
SEED = 1337

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = torch.nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj = torch.nn.Linear(N_EMBD, N_EMBD)

    def forward(self, x):
        batch, length, channels = x.shape
        query, key, value = self.c_attn(x).split(N_EMBD, dim=2)
        query = query.view(batch, length, N_HEAD, channels // N_HEAD).transpose(1, 2)
        key = key.view(batch, length, N_HEAD, channels // N_HEAD).transpose(1, 2)
        value = value.view(batch, length, N_HEAD, channels // N_HEAD).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
        y = y.transpose(1, 2).reshape(batch, length, channels)
        return self.c_proj(y)


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = torch.nn.Linear(N_EMBD, 4 * N_EMBD)
        self.c_proj = torch.nn.Linear(4 * N_EMBD, N_EMBD)

    def forward(self, x):
        return self.c_proj(torch.nn.functional.gelu(self.c_fc(x), approximate="tanh"))


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(N_EMBD)
        self.attn = Attention()
        self.ln_2 = torch.nn.LayerNorm(N_EMBD)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = torch.nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.wpe = torch.nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = torch.nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = torch.nn.LayerNorm(N_EMBD)
        self.lm_head = torch.nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx):
        batch, length = idx.shape
        positions = torch.arange(length, device=idx.device)
        x = self.wte(idx) + self.wpe(positions)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


torch.manual_seed(SEED)
prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "The largest animal on land is"

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode_ordinary(prompt)[-BLOCK_SIZE:-1] if len(enc.encode_ordinary(prompt)) >= BLOCK_SIZE else enc.encode_ordinary(prompt)
idx = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

state = torch.load(CKPT_FILE, map_location=DEVICE, weights_only=True)
model = GPT().to(DEVICE)
model.load_state_dict(state)
model.eval()

with torch.no_grad():
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        for _ in range(MAX_NEW_TOKENS):
            logits = model(idx[:, -BLOCK_SIZE:])
            probs = torch.softmax(logits[:, -1] / TEMPERATURE, dim=-1)
            probs,indices = torch.topk(probs, TOP_K)
            next_token = indices[0][torch.multinomial(probs[0], 1)]
            idx = torch.cat([idx, next_token.unsqueeze(0)], dim=1)
            if idx[0, -1].item() == enc.eot_token:
                break

print(enc.decode([
    t for t in idx[0].tolist() if t != enc.eot_token
]))
