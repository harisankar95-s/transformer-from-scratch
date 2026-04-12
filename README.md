# Transformer From Scratch
Complete transformer architecture implemented from scratch in PyTorch — no nn.Transformer, no pre-built attention modules, no shortcuts.

## What This Is
Every component of the original "Attention Is All You Need" paper, built from the ground up:

- Vocabulary indexing and word embeddings
- Positional encoding (sine/cosine)
- Multi-head self-attention with scaled dot-product
- Feed-forward network
- Add & Norm (residual connections + layer normalization)
- Full encoder stack (6 layers)
- Masked self-attention, cross-attention, full decoder stack (6 layers)
- Output linear projection + softmax

## Why
I wanted to go beyond using transformer APIs and actually understand what happens inside — every matrix multiplication, every dimension, every gradient flow. If you can build it from scratch, you truly understand it.

## Tech Stack
Python, PyTorch
