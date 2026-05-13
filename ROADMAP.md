# Tiny Darija charGPT - Step By Step Roadmap

**Last updated:** 2026-05-09  
**Goal:** Build a tiny Darija GPT-style chatbot in C++ from scratch.  
**Strategy:** CPU-only training and inference. GPU/CUDA is not part of this roadmap.

---

## Hardware Strategy

This project will use **CPU only**.

Current target machine:

```text
CPU: usable for tiny educational training
RAM: enough for a small model
GPU/CUDA: not required and not planned
```

This means the model should stay small and simple.

Recommended CPU-friendly starting values:

```text
context_length = 64
embedding_dim = 64
heads = 2
layers = 1
batch_size = 4 or 8
```

Avoid very large settings because CPU training will become slow.

---

## Current Project Status

Already done:

- [x] C++17 project with `Makefile`
- [x] Data file: `data/darija.txt`
- [x] Byte tokenizer files:
  - `src/ByteTokenizer.hpp`
  - `src/ByteTokenizer.cpp`
- [x] `src/main.cpp` loads and tokenizes the Darija dataset

Current files:

```text
llm/
├── Makefile
├── ROADMAP.md
├── data/
│   └── darija.txt
└── src/
    ├── ByteTokenizer.hpp
    ├── ByteTokenizer.cpp
    └── main.cpp
```

Run now:

```bash
make
./llm
```

---

# Full Step-By-Step Build Plan

## Phase 1: Tokenization

**Goal:** Convert text into numbers and numbers back into text.

You already started this phase.

Current tokenizer:

```text
0-255 = raw byte tokens
256   = end-of-text token
vocab_size = 257
```

Files:

- [x] `src/ByteTokenizer.hpp`
- [x] `src/ByteTokenizer.cpp`

Tasks:

- [x] Encode text to token IDs.
- [x] Decode token IDs back to text.
- [x] Add end-of-text token.

Checkpoint:

```text
Input:  salam 3likom
Output: token IDs
Decode: salam 3likom
```

---

## Phase 2: Dataset And Batches

**Goal:** Prepare training examples from `data/darija.txt`.

A language model learns like this:

```text
input tokens:  [s, a, l, a, m]
target tokens: [a, l, a, m,  ]
```

The target is always the next token.

Files to add:

- [ ] `src/Dataset.hpp`
- [ ] `src/Dataset.cpp`

Tasks:

- [ ] Load full text.
- [ ] Encode full text into tokens.
- [ ] Split data into train and validation.
- [ ] Create input/target pairs.
- [ ] Create random batches.
- [ ] Use a context length like `64` first.

First values:

```text
context_length = 64
batch_size = 8
```

Checkpoint:

```text
Print one batch:
input:  64 tokens
target: same 64 tokens shifted by one
```

---

## Phase 3: Embeddings

**Goal:** Convert token IDs into learnable vectors.

Token IDs are just numbers. A neural network needs vectors.

Example:

```text
token_id 97 -> [0.02, -0.01, 0.15, ...]
```

Files to add:

- [x] `src/Embedding.hpp`
- [x] `src/Embedding.cpp`

Tasks:

- [x] Create token embedding table.
- [x] Initialize weights randomly.
- [x] Lookup embedding vector for each token.

First values:

```text
vocab_size = 257
embedding_dim = 32
```

Checkpoint:

```text
Input token shape:      batch_size x context_length
Embedding output shape: batch_size x context_length x embedding_dim
```

---

## Phase 4: Position Embeddings

**Goal:** Teach the model token order.

Without position embeddings, the model knows which tokens exist but not where they appear.

Example:

```text
final_embedding = token_embedding + position_embedding
```

Files to update:

- [x] `src/Embedding.hpp`
- [x] `src/Embedding.cpp`

Tasks:

- [x] Create position embedding table.
- [x] One vector for each position from `0` to `context_length - 1`.
- [x] Add token embedding and position embedding together.

First values:

```text
context_length = 64
embedding_dim = 32
```

Checkpoint:

```text
Every token now has content information + position information.
```

---

## Phase 5: Linear Layer

**Goal:** Convert embeddings into prediction scores.

The model predicts the next token by producing one score for every possible token.

```text
embedding -> linear layer -> logits
```

Files to add:

- [x] `src/Linear.hpp`
- [x] `src/Linear.cpp`

Tasks:

- [x] Implement matrix multiplication.
- [x] Implement bias addition.
- [x] Convert embedding vectors into logits.

Shape:

```text
input:  embedding_dim
output: vocab_size = 257
```

Checkpoint:

```text
For each input token, the model outputs 257 scores.
```

---

## Phase 6: Softmax And Loss

**Goal:** Measure how wrong the model is.

Softmax converts scores into probabilities.

Cross-entropy loss tells us how bad the prediction was.

Files to add:

- [x] `src/Loss.hpp`
- [x] `src/Loss.cpp`

Tasks:

- [x] Implement softmax.
- [x] Implement cross-entropy loss.
- [x] Compare prediction with the target next token.
- [x] Print loss.

Important baseline:

```text
Random loss around log(257) = 5.55
```

Checkpoint:

```text
The program can say: loss = 5.5 or lower.
```

---

## Phase 7: Backpropagation And Optimizer

**Goal:** Make the model learn by updating weights.

Training loop:

```text
forward pass
calculate loss
backward pass
update weights
repeat
```

Files to add:

- [x] `src/Trainer.hpp`
- [x] `src/Trainer.cpp`

> Note: `Optimizer.hpp/cpp` was removed. Gradient updates are applied directly in each layer's `apply_gradients()` method.

Tasks:

- [ ] Compute gradients for linear layer.
- [ ] Compute gradients for embeddings.
- [ ] Implement simple SGD first.
- [ ] Print loss every few steps.
- [ ] Check that loss goes down.

First optimizer:

```text
SGD learning_rate = 0.01
```

Checkpoint:

```text
Loss starts high and becomes lower after training.
```

---

## Phase 8: Simple Text Generation

**Goal:** Generate Darija-like text from the trained simple model.

Generation loop:

```text
1. Give model current tokens.
2. Predict next token probabilities.
3. Sample one token.
4. Add it to the text.
5. Repeat.
```

Files to add:

- [ ] `src/Sampler.hpp`
- [ ] `src/Sampler.cpp`

Tasks:

- [ ] Sample from probabilities.
- [ ] Add temperature.
- [ ] Stop at EOT token or max length.
- [ ] Decode tokens into text.

Checkpoint:

```bash
./llm --generate 300
```

Expected result: messy but partly Darija-like text.

---

## Phase 9: Self-Attention

**Goal:** Let each token look at previous tokens.

Attention is the main idea inside GPT.

Formula:

```text
Q = X * Wq
K = X * Wk
V = X * Wv
scores = Q * K^T / sqrt(head_dim)
weights = softmax(scores with causal mask)
output = weights * V
```

Files to add:

- [x] `src/MultiHeadAttention.hpp`
- [x] `src/MultiHeadAttention.cpp`

> Note: Single-head `Attention.hpp/cpp` was merged into `MultiHeadAttention`.

Tasks:

- [x] Implement query projection.
- [x] Implement key projection.
- [x] Implement value projection.
- [x] Implement causal mask.
- [x] Implement attention scores.
- [x] Implement attention output.

First values:

```text
context_length = 64
embedding_dim = 128
heads = 4
```

Checkpoint:

```text
Each token can only attend to itself and previous tokens, not future tokens.
```

---

## Phase 10: Multi-Head Attention

**Goal:** Run multiple attention heads in parallel.

One head learns one type of relationship. Multiple heads can learn different relationships.

Files to update:

- [x] `src/MultiHeadAttention.hpp`
- [x] `src/MultiHeadAttention.cpp`

Tasks:

- [x] Split embedding dimension into heads.
- [x] Run attention per head.
- [x] Concatenate head outputs.
- [x] Add output projection.

First values:

```text
embedding_dim = 128
heads = 4
head_dim = 32
```

Checkpoint:

```text
Multi-head attention output has same shape as input embeddings.
```

---

## Phase 11: Feed-Forward Network

**Goal:** Add a small neural network after attention.

Transformer block uses:

```text
attention
feed-forward network
```

Files to add:

- [x] `src/FeedForward.hpp`
- [x] `src/FeedForward.cpp`

Tasks:

- [x] Linear layer from `embedding_dim` to `4 * embedding_dim`.
- [x] GELU activation.
- [x] Linear layer back to `embedding_dim`.

Shape:

```text
64 -> 256 -> 64
```

Checkpoint:

```text
Feed-forward network input and output shapes match.
```

---

## Phase 12: LayerNorm And Residual Connections

**Goal:** Make the Transformer stable during training.

Transformer block pattern:

```text
x = x + Attention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

Files to add:

- [x] `src/LayerNorm.hpp`
- [x] `src/LayerNorm.cpp`
- [x] `src/TransformerBlock.hpp`
- [x] `src/TransformerBlock.cpp`

Tasks:

- [x] Implement LayerNorm.
- [x] Add residual connection around attention.
- [x] Add residual connection around feed-forward network.

Checkpoint:

```text
Transformer block has stable forward pass.
```

---

## Phase 13: Transformer Block

**Goal:** Combine attention, feed-forward, LayerNorm, and residuals into one block.

Files to add:

- [x] `src/TransformerBlock.hpp`
- [x] `src/TransformerBlock.cpp`

Block structure:

```text
input
  -> LayerNorm
  -> Multi-Head Attention
  -> Residual Add
  -> LayerNorm
  -> Feed-Forward
  -> Residual Add
output
```

Checkpoint:

```text
Input shape and output shape are the same.
```

---

## Phase 14: Mini GPT Model

**Goal:** Build the full tiny GPT model.

Files to add:

- [x] `src/GPT.hpp`
- [x] `src/GPT.cpp`

Model structure:

```text
token IDs
  -> token embeddings
  -> position embeddings
  -> transformer block(s)
  -> final LayerNorm
  -> linear language head
  -> logits over vocab
```

Current model size:

```text
vocab_size = 257
context_length = 64
embedding_dim = 128
heads = 4
layers = 2
batch_size = 8
```

Checkpoint:

```text
Forward pass produces:
batch_size x context_length x vocab_size
```

---

## Phase 15: Train Mini GPT

**Goal:** Train the GPT model on Darija text.

Tasks:

- [x] Load dataset.
- [x] Create batches.
- [x] Forward pass.
- [x] Compute cross-entropy loss.
- [x] Backward pass.
- [x] Update weights.
- [x] Print train loss.
- [x] Print validation loss.
- [x] Generate sample text during training.

Targets:

```text
random loss: about 5.55
first good target: below 4.0
better target: below 3.0
excellent target: below 2.5 (requires more data / larger model)
```

Checkpoint:

```text
Training loss decreases over time.
```

---

## Phase 16: Save And Load Model

**Goal:** Save trained weights and load them later.

Files/directories to add:

- [x] `models/`
- [x] `models/darija_gpt.bin`

Save:

- [x] Model hyperparameters.
- [x] Tokenizer type: byte tokenizer.
- [x] Vocab size: `257`.
- [x] EOT token: `256`.
- [x] All weights.

Load:

- [x] Check file exists.
- [x] Check version/header.
- [x] Check dimensions match.
- [x] Load weights.

Checkpoint:

```text
Train -> save -> close program -> load -> generate text.
```

---

## Phase 17: Chat Mode

**Goal:** Use the model in a terminal conversation.

Files to add:

- [x] `src/chat.cpp`

Prompt format:

```text
User: salam
Bot:
```

Tasks:

- [x] Read user input.
- [x] Keep conversation history.
- [x] Truncate history to context length.
- [x] Generate bot response.
- [x] Support `/quit`.
- [x] Support temperature.
- [x] Support max generation length.

Checkpoint:

```bash
./chat models/darija_gpt.bin
```

---

## Phase 18: CPU Optimization And Cleanup

**Goal:** Make the CPU model easier to run, debug, and improve.

This project will stay CPU-only, so this phase focuses on simple performance and code quality improvements instead of CUDA.

CPU tasks:

- [ ] Keep model dimensions small enough for the available RAM.
- [ ] Add timing logs for training steps.
- [ ] Print tokens-per-second during training.
- [ ] Avoid unnecessary vector copies in hot loops.
- [ ] Use `-O2` or `-O3` compiler optimization.
- [ ] Keep a simple debug mode for checking shapes and loss.
- [ ] Save small checkpoints regularly.

Checkpoint:

```text
The tiny GPT trains on CPU with stable loss and understandable speed logs.
```

---

# Recommended Learning Order

```text
1. Tokenization
2. Dataset batches
3. Embeddings
4. Position embeddings
5. Linear layer
6. Softmax and loss
7. Backpropagation and optimizer
8. Text generation
9. Self-attention
10. Multi-head attention
11. Feed-forward network
12. LayerNorm and residual connections
13. Transformer block
14. Mini GPT
15. Training
16. Save/load
17. Chat mode
18. CPU optimization and cleanup
```

---

# How To Run

## Build

```bash
make
```

This creates two executables:
- `./llm`  - training and generation
- `./chat` - interactive chat mode

## Train

```bash
./llm
```

This will:
1. Load `data/darija.txt`
2. Train the GPT model for 10,000 steps (configurable in `src/main.cpp`)
3. Save weights to `models/darija_gpt.bin`
4. Generate sample text

## Chat

```bash
./chat models/darija_gpt.bin
```

Commands inside chat:
- `/quit` - exit
- `/temp 0.8` - set temperature
- `/len 100` - set max response length
- Type anything else to chat

## Clean Build

```bash
make clean && make
```
