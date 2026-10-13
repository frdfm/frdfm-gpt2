# GPT-2 From Scratch — Trained on FineWeb

A minimal, educational reimplementation of GPT-2 (the 124M configuration) in pure PyTorch. The whole model is ~90 lines in [train.py](train.py). This README explains it the way it was written: top to bottom, intuition first.

| File | Purpose |
|---|---|
| [prepare_data.py](prepare_data.py) | Downloads FineWeb, turns text into token numbers → `data/train.bin`, `data/val.bin` |
| [train.py](train.py) | The GPT-2 model + the training loop that teaches it |
| [gen.py](gen.py) | Loads the trained model and writes text |

## Quick start

```bash
pip install torch numpy datasets tiktoken
python prepare_data.py   # tokenizes 300M tokens into data/*.bin
python train.py          # logs to logs/train.log, saves checkpoints/model.pt
python gen.py "The largest animal on land is"
```

Needs a CUDA GPU. Token IDs fit in `uint16` because the vocab (50,257) is under 65,535.

## The one idea behind everything

GPT-2 has exactly one job: **given some text, guess what comes next.** Do this at internet scale — 300M tokens — and the *only* way to get good at it is to learn grammar, facts, and reasoning as side effects. The architecture below is just a very clever machine for making the next-token guess.

The model is defined by five numbers:

```python
VOCAB_SIZE = 50257   # all the word-pieces it knows
BLOCK_SIZE = 1024    # how much text it can look at at once
N_LAYER    = 12      # blocks in the stack
N_HEAD     = 12      # attention heads per block
N_EMBD     = 768     # width of every vector that flows through
```

## The forward pass, input → output

What happens to one batch of text inside `GPT.forward`:

```
idx (token IDs)                      [batch, 1024]        ints 0..50256
    │  STAGE 1 — each word gets a "what" and a "where"
x  residual stream                   [batch, 1024, 768]
    │  STAGE 2 — 12 blocks; the stream's shape never changes
    │      x = x + Attention(LayerNorm(x))   # tokens talk to each other
    │      x = x + MLP(LayerNorm(x))         # each token thinks alone
    ▼
x                                    [batch, 1024, 768]
    │  STAGE 3 — score every word it knows
logits                               [batch, 1024, 50257]
    │  STAGE 4 — penalize wrong guesses
loss                                 scalar
```

**Stage 1 — words become vectors.** Neural networks need numbers. Two lookup tables: `wte` answers *"what is this word?"* — each of the 50,257 tokens has a learned 768-dim meaning vector; `wpe` answers *"where am I in the sentence?"* — each position 0..1023 has one too. Add them: every token now carries both its meaning and its location in one vector. This tensor is the **residual stream** — imagine a whiteboard holding one line per token, which all later stages read from and scribble corrections onto.

**Stage 2 — 12 rounds of "talk, then think".** Each block does two things: first attention, which lets a token reach back into earlier tokens and pull in useful context ("we" wants to grab the person's name from six words ago); then an MLP, where each token digests that context on its own. Each block writes its updates back onto the same whiteboard — it never erases it — so 12 blocks means 12 increasingly deep interpretations: early blocks see spelling and phrases, late blocks see story logic. **The important consequence:** inserting something into a position's whiteboard must wait until attention is computed before it appears, which is why attention has a hard rule — *a token may only look backward*.

**Stage 3 — back to words.** `ln_f` tidies the whiteboard, then `lm_head` scores **every token in the vocabulary** at every position: `[batch, 1024, 50257]`. The scorer is literally the token-embedding table reused (`lm_head.weight = wte.weight` — "weight tying"): the score for token *v* is the dot product of the token's current summary with *v*'s meaning vector — "how well does each word fit where I just got to?" Everything is computed for *all* 1024 positions at once, so one forward pass yields 1,024 next-word guesses.

**Stage 4 — the exam.** `targets` is the same text shifted one slot left: the true next token at every position. Cross-entropy punishes guesses with `-log p(correct next token)` — a confident wrong answer costs far more than a humble one. Val loss (nats/token) is the score that goes in `logs/train.log`; lower is better.

### Inside one block

A block is just four small parts, wired as `x + Attention(LayerNorm(x))`, then `x + MLP(LayerNorm(x))`:

- The **`+` (residual connection)** is the "never erase the whiteboard" rule. Each part only ever *adds* a correction, so even block 12 can still improve on block 1's work and gradients can flow back anywhere.
- **LayerNorm** is a reset button before each part: it rescales a token's 768 numbers to mean 0 / unit variance (then re-scales with two learned knobs). Tokens entering a subsystem can't be "shouty" from an earlier correction.
- **Attention** — the talking part. Each token produces three projections (`c_attn: 768 → 2304`):
  - a **query** — "here's what I'm looking for",
  - a **key** — "here's what I contain",
  - a **value** — "here's what I'll hand over if you choose me".

  Every query is compared against every earlier key → a `1024 × 1024` relevance matrix (`query·keyᵀ`). Each row is scaled by `1/√64 = 8` so the comparison doesn't overwhelm the softmax, masked so only positions `≤ t` are visible (this is the no-looking-ahead rule), softmaxed into weights summing to 1, and each token receives a weighted blend of chosen values. This whole dance runs in **12 independent heads** — 12 specialists, each with its own 64-dim question: one may track which noun owns a verb, another which quotes are open. PyTorch does it with one fused call: `F.scaled_dot_product_attention(..., is_causal=True)`. Heads are concatenated and mixed by `c_proj: 768 → 768` before being added to the stream.
- **MLP** — the thinking part. Each token, alone: `768 → 3072` (`c_fc`) → **GELU** (a smooth gate most active when the input is relevant positive) → `768` (`c_proj`). Think of the 3072 middle units as pattern detectors activating to different degrees.

### Where the 124,439,808 parameters live

| Component | Shape | Params |
|---|---:|---:|
| `wte` (tied with `lm_head`) | [50257, 768] | 38,597,376 |
| `wpe` | [1024, 768] | 786,432 |
| Per block: attn `c_attn` / `c_proj` | 768/2304/768 | 1,771,776 + 590,592 |
| Per block: MLP `c_fc` / `c_proj` | 768/3072/768 | 2,362,368 + 2,360,064 |
| Per block: 2 LayerNorms | 2 × 1536 | 3,072 |
| **12 blocks** | | **85,054,464** |
| `ln_f` | | 1,536 |
| **Total** | | **124,439,808** |

Everything starts as small noise (`normal, std 0.02`); each `c_proj` is shrunk further by `1/√(2·N_LAYER) ≈ 1/4.9` so 24 back-to-back additions don't inflate the whiteboard out of numerical range before training starts.

## The data: a giant binary of numbers

`prepare_data.py` streams **FineWeb** (whole web pages), chops text into subword pieces with the GPT-2 tokenizer, marks the end of each document with the `eot` token, and packs all of it into one flat file of 16-bit ints — 299,000,000 for training, 1,000,000 for testing. `train.py` streams it back with `np.memmap`, reading only the 8 KB windows it needs.

Each training batch is one **contiguous 8,193-token slice** of that file, shaped into 8 rows of 1,024:

```
file:   ... a  b  c  d  e  f  g  h ...
input  x:  a  b  c  d  e  f  g       target y:  b  c  d  e  f  g  h
```

The very same numbers serve as inputs *and* grade sheet — shifted by one. No shuffling, no labels to build; every step reads a fresh slice and consumes the whole dataset exactly once.

## The training: 570 nudges in the right direction

Each step: forward (bf16, tensor cores) → backward (accumulate gradients over 64 micro-batches of 8) → drop the norm of all gradients at once to ≤ 1.0 (a "fuse" against rare ill-conditioned spikes) → AdamW updates weights with a direction/size per parameter: `lr` 6e-4, smoothed gradient `betas (0.9, 0.95)`, indivisible floor `eps` 1e-8, `weight_decay` 0.1 gently pulling weights toward zero ("less sure, less committed").

The **learning rate** isn't constant — that's the trick most of the recipe:

- **Steps 0–199: linear warmup.** At the start the model is noise and the gradient is unreliable; a full-size step would scatter it. So the lr climbs from ~0 to 6e-4.
- **Steps 200–570: cosine decay to 6e-5.** As the solution forms, big steps bounce around a minimum instead of settling. The lr glides down, like easing the accelerator to park crisply in the spot.

Big effective batch: 8 × 64 = **512 sequences = 524,288 tokens per step**, completed over `(299M − 1) / 524,288 = 570` steps. Every 10 steps, the mean val loss over 81,920 tokens is logged and `checkpoints/model.pt` is overwritten.

## Generation (`gen.py`)

Trained on next-token prediction, generation is embarrassingly simple: feed text → take **only the last position's** scores → softmax to probabilities. Two dials keep it tame: **temperature 0.8** (divides the logits; <1 favors the likely words) and **top-k 40** (keep only the 40 best, drop the tail of gibberish candidates). Sample, append, repeat — up to 100 tokens, stop at the `eot` token. A fixed seed (1337) makes a prompt reproducible.

## References

- Radford et al., *Language Models are Unsupervised Multitask Learners* (2019) — GPT-2
- Vaswani et al., *Attention Is All You Need* (2017) — Transformer
- Penedo et al., *The FineWeb Datasets* (2024) — the corpus
