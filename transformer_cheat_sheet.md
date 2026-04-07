# Transformer Cheat Sheet

A plain-English reference for transformer concepts, built up while working on `mech_interp_visuals`.

---

## Vocabulary

**Token** — A piece of input text. Could be a whole word, a subword chunk, punctuation, or a character. The model never sees raw text — it sees token IDs (integers) that index into a vocabulary.

**Vocabulary (d_vocab)** — The full set of tokens the model knows about. GPT-2 has 50,257.

---

## Dimensions

**d_model** — The width of the residual stream. Every token is represented as a vector of this size at every layer. GPT-2 small: 768.

**n_layers** — Number of transformer blocks stacked on top of each other. GPT-2 small: 12.

**n_heads** — Number of attention heads per layer. GPT-2 small: 12.

**d_head** — The dimension each attention head works in. Usually d_model / n_heads. GPT-2 small: 64.

**d_mlp** — The hidden dimension of the MLP block. Usually 4× d_model. GPT-2 small: 3072.

**n_ctx** — Context length. The maximum number of tokens the model can process at once. GPT-2 small: 1024.

---

## Components

### Residual Stream
The d_model-dimensional vector at each token position that flows through the entire network. Every component (attention, MLP) reads from it and adds its output back to it — it's never replaced, only accumulated. Think of it as the I/O bus between layers.

### Token Embedding (Embed)
A lookup table of shape `d_vocab × d_model`. Maps each token ID to a learned d_model-dimensional vector.

### Positional Embedding (Pos Embed)
A lookup table of shape `n_ctx × d_model`. One learned vector per position (0 through n_ctx-1). Added to the token embedding so the model knows word order. Same shape as the token embedding — just indexed by position instead of token ID.

### Attention Heads
Each layer has n_heads parallel attention heads. Each head:
1. Projects the residual stream into queries (Q), keys (K), and values (V) using learned weight matrices W_Q, W_K, W_V (each d_model × d_head)
2. Computes attention scores: QKᵀ / √d_head → softmax → attention pattern A
3. Reads from values: A · V
4. Projects back up to d_model with W_O (d_head × d_model)
5. Adds result to the residual stream via ⊕

The attention pattern A tells each token position how much to "attend to" (borrow information from) each other position.

### MLP (Feed-Forward Network)
Applied after attention in each layer. Two linear projections with a non-linear activation in between:

```
x → W_in (d_model × d_mlp) → GeLU → W_out (d_mlp × d_model) → y
```

Expands to d_mlp (4× wider), applies the activation, then contracts back. Thought to store factual knowledge. Output is added to the residual stream via ⊕.

**W_in dimensions**: d_model × d_mlp (e.g., 768 × 3072 for GPT-2 small)
**W_out dimensions**: d_mlp × d_model (e.g., 3072 × 768 for GPT-2 small)

### LayerNorm (LN)
Applied before attention and before MLP in each layer. Two steps:
1. **Normalize**: rescale the d_model-dim vector to have mean=0, variance=1 across its dimensions
2. **Rescale**: apply a learned scale γ and bias β (both size d_model) to restore useful signal

**Why normalize at all?** During training, activations (the numbers flowing through the network) can compound across layers — matrix multiplications can cause values to grow very large or shrink toward zero. When that happens, gradients (the training signal) either explode or vanish, making training unstable or slow. LayerNorm reins values back into a stable range at each layer.

**Why not normalize completely?** Hard normalization (mean=0, var=1, always) throws away information — the model may have learned to encode meaning in the magnitude or offset of activations. The learned γ and β let the model say "normalize for stability, but then shift/scale back to whatever range I actually need." γ and β are initialized to 1 and 0 (identity) and learned during training. They're cheap (2 × d_model parameters) compared to the rest of the model.

### Unembedding (Unembed)
A linear projection from d_model → d_vocab. Takes the final residual stream vector for each position and produces logits over the vocabulary.

### Logits
The raw output scores over the vocabulary (d_vocab numbers per position). Apply softmax to convert to probabilities. The highest-probability token is the model's predicted next token.

### ⊕ (Addition)
Not a learned component — just addition. The output of each attention/MLP block is added to the residual stream rather than replacing it. This is the "residual" in residual stream.

---

## Color Coding (in the visualization)

| Color | Meaning | Components |
|-------|---------|------------|
| Tan | Learned parameters (has trained weights) | Embed, Pos Embed, LN, Attention heads, MLP, Unembed |
| Gray | I/O — data flowing through the system | Tokens (input), Logits (output), Residual stream (between layers) |
| White | Pure operations (no parameters) | ⊕ addition, Softmax, GeLU/ReLU |

---

## Operations (inside blocks)

**Softmax** — Converts a vector of raw scores into a probability distribution (all values 0–1, sum to 1). Used in attention to convert QKᵀ scores into attention weights. No learned parameters.

**GeLU / ReLU** — Non-linear activation functions applied in the MLP between W_in and W_out. Without non-linearity, stacking linear layers would collapse to a single linear transformation. No learned parameters.
