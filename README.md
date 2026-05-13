# Darija GPT

A tiny GPT-style language model built from scratch in C++17, trained on Moroccan Darija (Moroccan Arabic) text. No PyTorch, no TensorFlow — just raw C++ with a custom autograd engine.

![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What is this?

This project implements a complete transformer-based language model (GPT architecture) entirely in C++ from the ground up. It includes:

- **Custom autograd engine** with forward/backward passes
- **Byte-level tokenizer** that handles any language natively
- **Multi-head self-attention** with causal masking
- **Transformer blocks** with LayerNorm and residual connections
- **GELU activation** and feed-forward networks
- **Cross-entropy loss** with numerically stable softmax
- **SGD optimizer** with gradient clipping
- **Text generation** with temperature, top-k, and repetition penalty
- **Model save/load** in a custom binary format
- **Interactive chat mode**

### Architecture

```
Input Tokens
    |
    v
+---------------+
|  Embedding    |  Token embeddings + Position embeddings
+---------------+
    |
    v
+---------------+     +---------------+
| Transformer   | --> | Transformer   | --> ... (x N layers)
| Block 1       |     | Block 2       |
+---------------+     +---------------+
    |
    v
+---------------+
|  LayerNorm    |
+---------------+
    |
    v
+---------------+
|  Linear (LM   |  Language modeling head
|    Head)      |
+---------------+
    |
    v
  Logits -> Softmax -> Next Token Prediction
```

---

## Quick Start

### Prerequisites

- C++17 compatible compiler (g++ or clang++)
- `make`
- A text file with Darija (or any) text at `data/darija.txt`

### Build

```bash
make
```

This creates two executables:
- `./llm` — Training and generation
- `./chat` — Interactive chat mode

### Train

```bash
./llm
```

The model will:
1. Load `data/darija.txt`
2. Train for 10,000 steps (default, configurable in `src/main.cpp`)
3. Save weights to `models/darija_gpt.bin`
4. Generate sample text from prompts

### Chat

```bash
./chat models/darija_gpt.bin
```

Commands:
- `/quit` — Exit
- `/temp 0.8` — Set temperature (creativity)
- `/len 100` — Set max response length
- Type anything else to chat!

---

## Model Configuration

Default hyperparameters (easily tweakable in `src/main.cpp`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | 257 | Bytes 0-255 + EOT token |
| `context_length` | 64 | Max sequence length |
| `embed_dim` | 128 | Embedding dimension |
| `num_heads` | 4 | Attention heads |
| `num_layers` | 2 | Transformer blocks |
| `batch_size` | 8 | Training batch size |
| `learning_rate` | 0.01 | Initial SGD learning rate |
| `lr_decay` | 0.995 | Learning rate decay |
| `training_steps` | 10000 | Total training steps |

**Expected loss targets:**
- Random baseline: `log(257) ≈ 5.55`
- Good model: `< 4.0`
- Better model: `< 3.0`

---

## Project Structure

```
llm/
├── Makefile                  # Build system
├── README.md                 # This file
├── explain.md               # Detailed educational explanation
├── ROADMAP.md               # Step-by-step build plan
├── data/
│   └── darija.txt           # Training corpus
├── models/
│   └── darija_gpt.bin       # Saved model weights
└── src/
    ├── main.cpp             # Training & generation program
    ├── chat.cpp             # Interactive chat program
    ├── GPT.hpp/cpp          # Full GPT model
    ├── TransformerBlock.hpp/cpp  # Transformer block
    ├── MultiHeadAttention.hpp/cpp # Multi-head attention
    ├── FeedForward.hpp/cpp  # FFN with GELU
    ├── LayerNorm.hpp/cpp    # Layer normalization
    ├── Linear.hpp/cpp       # Dense layer
    ├── Embedding.hpp/cpp    # Token + position embeddings
    ├── Loss.hpp/cpp         # Cross-entropy loss
    ├── Dataset.hpp/cpp      # Batch generation
    ├── Sampler.hpp/cpp      # Temperature/top-k sampling
    ├── ByteTokenizer.hpp/cpp # Byte-level tokenizer
    └── Trainer.hpp/cpp      # Training loop
```

---

## How It Works

### 1. Tokenization

Uses byte-level tokenization — every character is its raw byte value (0-255). This means:
- **No vocabulary training needed** — works with any language immediately
- **Handles Darija natively** — Arabic letters, Latin letters, numbers, emojis
- **Vocabulary size: 257** (256 bytes + 1 EOT token)

```cpp
ByteTokenizer tokenizer;
auto tokens = tokenizer.encode("salam 3likom");
// Result: [115, 97, 108, 97, 109, 32, 51, 108, 105, 107, 111, 109]
```

### 2. Training Loop

```cpp
for each batch:
    1. Zero gradients
    2. Forward pass:  tokens -> embeddings -> attention -> FFN -> logits
    3. Compute loss:  cross_entropy(logits, targets)
    4. Backward pass: compute gradients for all weights
    5. Update weights: SGD with gradient clipping
    6. Decay learning rate
```

### 3. Text Generation

Autoregressive generation — predict one token at a time:

```cpp
auto generated = model.generate(prompt_tokens, max_length, temperature);
```

- **Temperature** controls randomness (0 = greedy, 0.8 = balanced, >1 = creative)
- **Top-k sampling** limits choices to the k most likely tokens
- **Repetition penalty** discourages repeating recent tokens

---

## Key Features

### Gradient Clipping

All layers implement per-layer L2 norm gradient clipping (`max_norm = 1.0`) to prevent exploding gradients during transformer training.

### Numerically Stable Softmax

Subtracts the maximum logit before `exp()` to avoid overflow:

```cpp
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

### Causal Masking

Attention only looks at previous tokens (not future ones), ensuring the model learns to predict rather than memorize:

```
         t0    t1    t2    t3
t0     [1.0,  0.0,  0.0,  0.0]
t1     [0.3,  0.7,  0.0,  0.0]
t2     [0.1,  0.2,  0.7,  0.0]
t3     [0.05, 0.05, 0.1,  0.8]
```

### Model Persistence

Weights are saved in a custom binary format with a magic header and version check:

```
File format:
  [8 bytes] "DARIJGPT" magic
  [4 bytes] version (1)
  [20 bytes] hyperparameters (vocab, context, embed, heads, layers)
  [...]     all weights sequentially
```

---

## Customization

### Change Model Size

Edit `src/main.cpp`:

```cpp
const int embed_dim = 256;      // Larger = more capable but slower
const int num_heads = 8;        // Must divide embed_dim evenly
const int num_layers = 4;       // More layers = deeper model
const int context_length = 128; // Longer memory
```

### Use Your Own Data

Replace `data/darija.txt` with any `.txt` file. The byte tokenizer handles any language automatically.

### Adjust Training

```cpp
const float learning_rate = 0.005f;  // Lower = more stable but slower
const int training_steps = 50000;    // More steps = better convergence
const int batch_size = 16;           // Larger = smoother gradients
```

---

## Performance Notes

This implementation uses `std::vector` for tensors, which is **educational but not optimal** for production. For a real deployment, you would want to:

- Use flat 1D arrays with custom indexing for better cache locality
- Add SIMD optimizations (AVX2/AVX-512)
- Consider a matrix library like Eigen or a lightweight BLAS
- For serious training: port to CUDA or use an existing framework

That said, on a modern CPU, this tiny model trains at a reasonable speed for educational purposes.

---

## Learning Resources

This codebase is designed to be read and understood. Check out:

- **`explain.md`** — A 1700+ line detailed explanation of every phase with concrete examples
- **`ROADMAP.md`** — Step-by-step build plan from scratch
- **Source code** — Heavily commented, clean C++17

Each component is self-contained:
- `Linear` = matrix multiplication + bias
- `Embedding` = lookup table
- `MultiHeadAttention` = the core transformer operation
- `LayerNorm` = normalization
- `FeedForward` = two linear layers with GELU
- `TransformerBlock` = wires everything together
- `GPT` = the full model
- `Trainer` = the training loop
- `Loss` = cross-entropy
- `Sampler` = generation strategies

---

## Troubleshooting

### Build fails with "cannot open data file"

Make sure `data/darija.txt` exists. Create it:

```bash
mkdir -p data
echo "salam 3likom" > data/darija.txt
```

### Loss stays at ~5.5 (random baseline)

- Check that your data file has enough text (at least a few thousand characters)
- Verify learning rate isn't too small
- Increase training steps

### Generated text is gibberish

This is expected for a tiny model! Try:
- Increasing `embed_dim` to 256+
- Adding more layers (`num_layers = 4`)
- Training for more steps
- Using a larger dataset

### Out of memory

Reduce model size:
```cpp
const int embed_dim = 64;
const int num_layers = 1;
const int batch_size = 4;
```

---

## License

MIT License — feel free to use, modify, and distribute.

---

## Acknowledgments

Built from scratch as an educational exercise in understanding how GPT and transformer models work at the lowest level. Inspired by:

- Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [llm.c](https://github.com/karpathy/llm.c)
- "Attention Is All You Need" (Vaswani et al., 2017)
- The GPT architecture (Radford et al., OpenAI)

---

**Happy coding!** If you find bugs or have questions, open an issue.
