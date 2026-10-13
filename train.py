import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_BIN = os.path.join(HERE, "data", "train.bin")
VAL_BIN = os.path.join(HERE, "data", "val.bin")
LOG_DIR = os.path.join(HERE, "logs")
CHECKPOINT_DIR = os.path.join(HERE, "checkpoints")
LOG_FILE = os.path.join(LOG_DIR, "train.log")
CKPT_FILE = os.path.join(CHECKPOINT_DIR, "model.pt")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

VOCAB_SIZE = 50257
BLOCK_SIZE = 1024
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768

BATCH_SIZE = 8
ACCUM_STEPS = 64
EVAL_BATCHES = 10
EVAL_INTERVAL = 10
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)

    def forward(self, x):
        batch, length, channels = x.shape
        query, key, value = self.c_attn(x).split(N_EMBD, dim=2)
        query = query.view(batch, length, N_HEAD, channels // N_HEAD).transpose(1, 2)
        key = key.view(batch, length, N_HEAD, channels // N_HEAD).transpose(1, 2)
        value = value.view(batch, length, N_HEAD, channels // N_HEAD).transpose(1, 2)
        y = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        y = y.transpose(1, 2).reshape(batch, length, channels)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.c_proj = nn.Linear(4 * N_EMBD, N_EMBD)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = Attention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self.init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, std=0.02 / (2 * N_LAYER) ** 0.5)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        batch, length = idx.shape
        positions = torch.arange(length, device=idx.device)
        x = self.wte(idx) + self.wpe(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def get_lr(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
    return LEARNING_RATE * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))


def get_batch(data, start):
    chunk = torch.from_numpy(data[start : start + tokens_per_batch + 1].astype(np.int64))
    x = chunk[:-1].view(BATCH_SIZE, BLOCK_SIZE).to(DEVICE)
    y = chunk[1:].view(BATCH_SIZE, BLOCK_SIZE).to(DEVICE)
    return x, y


def evaluate(model):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(EVAL_BATCHES):
            x, y = get_batch(val_data, i * tokens_per_batch)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                _, loss = model(x, y)
            total += loss.item()
    model.train()
    return total / EVAL_BATCHES


if __name__ == "__main__":
    train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    tokens_per_batch = BATCH_SIZE * BLOCK_SIZE
    total_steps = (len(train_data) - 1) // (tokens_per_batch * ACCUM_STEPS)
    WARMUP_STEPS = 200
    model = GPT().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8, weight_decay=WEIGHT_DECAY)
    print(f"parameters {sum(p.numel() for p in model.parameters()):,}")
    print(f"steps {total_steps}")

    log = open(LOG_FILE, "w")
    log.write("step train_loss val_loss lr seconds\n")
    start_time = time.time()

    for step in range(total_steps):
        optimizer.param_groups[0]["lr"] = get_lr(step)
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro in range(ACCUM_STEPS):
            x, y = get_batch(train_data, (step * ACCUM_STEPS + micro) * tokens_per_batch)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            (loss / ACCUM_STEPS).backward()
            accum_loss += loss.item() / ACCUM_STEPS

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if (step + 1) % EVAL_INTERVAL == 0 or step + 1 == total_steps:
            val_loss = evaluate(model)
            line = f"{step + 1} {accum_loss:.4f} {val_loss:.4f} {get_lr(step):.2e} {time.time() - start_time:.0f}"
            print(line)
            log.write(line + "\n")
            log.flush()
            torch.save(model.state_dict(), CKPT_FILE)

    log.close()
