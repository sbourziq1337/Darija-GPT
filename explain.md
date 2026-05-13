# Tiny Darija GPT - Complete Explanation (Phases 1-17)

This document explains every phase of building the Tiny Darija GPT model from scratch, using concrete examples and visual tables.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Phase 1: Tokenization](#phase-1-tokenization)
3. [Phase 2: Dataset and Batches](#phase-2-dataset-and-batches)
4. [Phase 3: Embeddings](#phase-3-embeddings)
5. [Phase 4: Position Embeddings](#phase-4-position-embeddings)
6. [Phase 5: Linear Layer](#phase-5-linear-layer)
7. [Phase 6: Softmax and Loss](#phase-6-softmax-and-loss)
8. [Phase 7: Backpropagation and Optimizer](#phase-7-backpropagation-and-optimizer)
9. [Phase 8: Text Generation](#phase-8-simple-text-generation)
10. [Phase 9: Self-Attention](#phase-9-self-attention)
11. [Phase 10: Multi-Head Attention](#phase-10-multi-head-attention)
12. [Phase 11: Feed-Forward Network](#phase-11-feed-forward-network)
13. [Phase 12: LayerNorm and TransformerBlock](#phase-12-layernorm-and-transformerblock)
14. [Updated File-by-File Reference](#updated-file-by-file-reference)

---

## The Big Picture

A language model learns one simple rule: **given what you have seen, predict what comes next.**

If you read enough Darija text, you start to notice patterns:
- After "salam", the next word is often "3likom"
- After "kifach", the next word is often "7alik"

A neural network learns these patterns by looking at millions of examples and adjusting its internal parameters.

### The Training Pipeline (What Happens Step by Step)

```
1. Raw text (darija.txt)
        |
        v
2. ByteTokenizer converts text into numbers (token IDs)
        |
        v
3. Dataset creates batches of input/target pairs
        |
        v
4. Embedding converts each token ID into a vector (list of numbers)
        |
        v
5. Multi-Head Attention mixes information between tokens
        |
        v
6. Feed-Forward Network processes each token independently
        |
        v
7. Linear layer converts vectors into prediction scores (logits)
        |
        v
8. Loss measures how wrong the predictions are
        |
        v
7. Backpropagation computes how to fix every weight
        |
        v
8. Optimizer updates the weights
        |
        v
   Repeat thousands of times --> model gets better
```

### Concrete Example

**Input text:** `"salam 3likom"`

**What the model sees:**
```
Input:  [s, a, l, a, m,  , 3, l, i, k, o, m]
Target: [a, l, a, m,  , 3, l, i, k, o, m, EOT]
```

At every position, the model must predict the **next** token.

---

## Phase 1: Tokenization

### Goal
Convert text into numbers and numbers back into text.

### Files
- `src/ByteTokenizer.hpp`
- `src/ByteTokenizer.cpp`

### The Problem
Computers only understand numbers, not letters. We need a way to convert every character in the text into a unique number.

### The Solution: Byte-Level Tokenization

Instead of creating a complex word vocabulary, we use the simplest possible approach: **bytes**.

Every character in your computer is stored as a byte (a number from 0 to 255).
- The letter `'a'` is byte `97`
- The letter `'b'` is byte `98`
- Arabic letter `'ا'` (alef) is stored as multiple bytes in UTF-8

We also add one special token:
- `256` = `<|endoftext|>` (EOT) - marks the end of a text

**Vocabulary size: 257** (bytes 0-255 + EOT)

### Concrete Example

**Text:** `"salam"`

**Encoding:**
```
's' -> 115
'a' -> 97
'l' -> 108
'a' -> 97
'm' -> 109

Result: [115, 97, 108, 97, 109]
```

**With EOT:**
```
[115, 97, 108, 97, 109, 256]
```

**Decoding:**
```
[115, 97, 108, 97, 109] -> "salam"
```

### Why This is Smart

- **No vocabulary training needed.** Works with any language immediately.
- **Handles Darija natively.** Arabic letters, Latin letters, numbers, emojis - everything is just bytes.
- **Simple and fast.** Direct mapping, no lookups needed.

### Tokenizer Functions

| Function | Input | Output | What it does |
|---|---|---|---|
| `encode(text)` | `"salam"` | `[115, 97, 108, 97, 109]` | Converts each character to its byte value |
| `decode(tokens)` | `[115, 97, 108, 97, 109]` | `"salam"` | Converts byte values back to characters |
| `encode_with_eot(text)` | `"salam"` | `[115, 97, 108, 97, 109, 256]` | Same as encode but adds EOT token at the end |
| `vocab_size()` | nothing | `257` | Returns total number of tokens |

---

## Phase 2: Dataset and Batches

### Goal
Prepare training examples from the tokenized text.

### Files
- `src/Dataset.hpp`
- `src/Dataset.cpp`

### The Problem
We have a long list of token IDs. We need to create many small training examples that the model can learn from.

### The Solution

**Step 1: Split into Train and Validation**

Take all tokens and split them:
- **90%** for training (the model learns from these)
- **10%** for validation (we use these to check if the model is overfitting)

**Step 2: Create Random Batches**

For each batch:
1. Pick a random starting position in the text
2. Take `block_size` (64) tokens as the **input**
3. Take the same `block_size` tokens but shifted by 1 as the **target**

### Concrete Example

**Full tokenized text:** `[115, 97, 108, 97, 109, 32, 51, 108, 105, 107, 111, 109, 256, ...]`

(Which is: "salam 3likom<EOT>...")

**Batch size = 2, Block size = 4**

**Example 1 - starting at position 0:**
```
Input:  [115, 97, 108, 97]     -> "sala"
Target: [97, 108, 97, 109]     -> "alam"
```

At each position, the target is the NEXT token:
- Given [115] predict 97
- Given [115, 97] predict 108
- Given [115, 97, 108] predict 97
- Given [115, 97, 108, 97] predict 109

**Example 2 - starting at position 3:**
```
Input:  [97, 109, 32, 51]      -> "am 3"
Target: [109, 32, 51, 108]     -> "m 3l"
```

**A full batch (2 sequences):**
```
Batch inputs:
  Sequence 0: [115, 97, 108, 97]
  Sequence 1: [97, 109, 32, 51]

Batch targets:
  Sequence 0: [97, 108, 97, 109]
  Sequence 1: [109, 32, 51, 108]
```

### Why Random Batches?

If we always train on the same examples in the same order, the model memorizes the order instead of learning the patterns. Random batches make the model robust.

### Dataset Functions

| Function | What it does |
|---|---|
| `Dataset(tokens, batch_size, block_size)` | Splits tokens into train (90%) and validation (10%) |
| `get_batch(Split::Train)` | Returns a random batch from the training set |
| `get_batch(Split::Validation)` | Returns a random batch from the validation set |

---

## Phase 3: Embeddings

### Goal
Convert token IDs into learnable vectors.

### Files
- `src/Embedding.hpp`
- `src/Embedding.cpp`

### The Problem
Neural networks work with numbers, not token IDs. A token ID like `97` is just an index. We need to convert it into a meaningful list of numbers that the network can process.

### The Solution: Token Embedding Table

We create a **lookup table** (a dictionary) where each token ID maps to a vector of `embedding_dim` numbers.

**Analogy:** Imagine a dictionary where each word has a personality profile:
```
Word     -> [happy, big, furry, fast]
"cat"    -> [0.8,   0.2,  0.9,  0.7]
"dog"    -> [0.9,   0.5,  0.8,  0.6]
"car"    -> [0.0,   0.7,  0.0,  0.9]
```

In our code, the "words" are token IDs (0-256) and the "personality" has 128 dimensions.

### Concrete Example

**Parameters:**
- `vocab_size = 257`
- `embedding_dim = 4` (using 4 instead of 32 for simplicity)

**Token Embedding Table (simplified):**
```
Token ID 97 ('a'):  [0.02, -0.01,  0.15,  0.03]
Token ID 108 ('l'): [0.05,  0.03, -0.02, -0.01]
Token ID 109 ('m'): [-0.01, 0.02,  0.08,  0.04]
Token ID 115 ('s'): [0.03, -0.02,  0.01,  0.06]
```

**Forward pass for input [115, 97, 108, 97] ("sala"):**
```
Output[0] = token_embedding[115] = [0.03, -0.02, 0.01, 0.06]  <- 's'
Output[1] = token_embedding[97]  = [0.02, -0.01, 0.15, 0.03]  <- 'a'
Output[2] = token_embedding[108] = [0.05,  0.03, -0.02, -0.01] <- 'l'
Output[3] = token_embedding[97]  = [0.02, -0.01, 0.15, 0.03]  <- 'a'
```

**Shape transformation:**
```
Input:  [115, 97, 108, 97]              (4 token IDs)
            |
            v
Output: [[0.03, -0.02, 0.01, 0.06],    (4 vectors of 4 numbers each)
         [0.02, -0.01, 0.15, 0.03],
         [0.05,  0.03, -0.02, -0.01],
         [0.02, -0.01, 0.15, 0.03]]
```

### Why Vectors?

Similar words should have similar vectors. During training, the model learns that:
- `'a'` and `'e'` (both vowels) should have similar vectors
- `'s'` and `'z'` (both sibilants) should have similar vectors

This allows the model to generalize. If it learns a pattern with `'a'`, it might also apply it to `'e'` because their vectors are similar.

---

## Phase 4: Position Embeddings

### Goal
Teach the model token order.

### Files
- Same as Phase 3: `src/Embedding.hpp` and `src/Embedding.cpp`

### The Problem
Without position information, the model only knows **which tokens exist**, not **where they appear**.

For example, these two sentences would look identical:
- "dog bites man"
- "man bites dog"

Both have the same tokens, just in different orders. The model needs to know the difference!

### The Solution: Position Embedding Table

We create a **second lookup table** where each **position** (0, 1, 2, 3, ...) maps to a vector.

**Analogy:** In a theater, every seat has a different view:
- Seat 0 (front row): close to the stage
- Seat 1 (second row): slightly further
- Each seat adds a unique "perspective" to the show

**Position Embedding Table (simplified, context_length = 4):**
```
Position 0: [0.01,  0.02, -0.01,  0.03]
Position 1: [0.03, -0.01,  0.02, -0.02]
Position 2: [-0.02, 0.03,  0.01,  0.01]
Position 3: [0.02,  0.01, -0.03,  0.02]
```

### How Position Embeddings Work

The **final embedding** for each token is:
```
final_embedding = token_embedding + position_embedding
```

**Concrete Example:**

Input: `[115, 97, 108, 97]` ("sala")

```
Position 0 ('s'):
  token_embedding[115] = [0.03, -0.02, 0.01, 0.06]
  position_embedding[0] = [0.01,  0.02, -0.01, 0.03]
  final = [0.04, 0.00, 0.00, 0.09]

Position 1 ('a'):
  token_embedding[97] = [0.02, -0.01, 0.15, 0.03]
  position_embedding[1] = [0.03, -0.01, 0.02, -0.02]
  final = [0.05, -0.02, 0.17, 0.01]

Position 2 ('l'):
  token_embedding[108] = [0.05, 0.03, -0.02, -0.01]
  position_embedding[2] = [-0.02, 0.03, 0.01, 0.01]
  final = [0.03, 0.06, -0.01, 0.00]

Position 3 ('a'):
  token_embedding[97] = [0.02, -0.01, 0.15, 0.03]
  position_embedding[3] = [0.02, 0.01, -0.03, 0.02]
  final = [0.04, 0.00, 0.12, 0.05]
```

**Key insight:** The same token `'a'` appears at positions 1 and 3, but gets **different final vectors** because the position embeddings are different!

---

## Phase 5: Linear Layer

### Goal
Convert embeddings into prediction scores.

### Files
- `src/Linear.hpp`
- `src/Linear.cpp`

### The Problem
We have vectors representing tokens. We need to predict which token comes next. We need one score for every possible token in the vocabulary.

### The Solution: Matrix Multiplication + Bias

The Linear layer is like a **voting committee**:
- You have `input_dim` advisors (each dimension of the embedding vector)
- You want to make decisions about `output_dim` possible outcomes (each token in the vocabulary)
- Each advisor has a different weight for each outcome

**Formula:**
```
output = (input * weights) + bias
```

### Concrete Example with Words

**Setup:**
- `input_dim = 4` (embedding dimension)
- `output_dim = 4` (vocabulary: "I", "can", "go", "alone")

**Weight Matrix** (4 advisors x 4 predictions):
```
                  Predict "I"   Predict "can"   Predict "go"   Predict "alone"
From feature 0:      0.01         -0.02           0.03           -0.01
From feature 1:     -0.01          0.03          -0.02            0.02
From feature 2:      0.02         -0.01           0.01           -0.03
From feature 3:     -0.02          0.02           0.02            0.01
```

**Bias** (initial preference for each word):
```
"I": 0.0, "can": 0.0, "go": 0.0, "alone": 0.0
```

**Input vector** (embedding for "I"):
```
[0.5, 0.1, -0.2, 0.3]
```

**Forward Pass Calculation:**

Predict "I":
```
score = 0.0 + (0.5*0.01) + (0.1*-0.01) + (-0.2*0.02) + (0.3*-0.02)
      = 0.0 + 0.005 - 0.001 - 0.004 - 0.006
      = -0.006
```

Predict "can":
```
score = 0.0 + (0.5*-0.02) + (0.1*0.03) + (-0.2*-0.01) + (0.3*0.02)
      = 0.0 - 0.01 + 0.003 + 0.002 + 0.006
      = 0.001
```

Predict "go":
```
score = 0.0 + (0.5*0.03) + (0.1*-0.02) + (-0.2*0.01) + (0.3*0.02)
      = 0.0 + 0.015 - 0.002 - 0.002 + 0.006
      = 0.017
```

Predict "alone":
```
score = 0.0 + (0.5*-0.01) + (0.1*0.02) + (-0.2*-0.03) + (0.3*0.01)
      = 0.0 - 0.005 + 0.002 + 0.006 + 0.003
      = 0.006
```

**Output logits:**
```
["I": -0.006, "can": 0.001, "go": 0.017, "alone": 0.006]
```

The model thinks "go" is the most likely next word (score 0.017).

### Full Prediction Table

For a sentence "I can go alone", the model produces scores for every position:

| Current Word | Score "I" | Score "can" | Score "go" | Score "alone" |
|:---:|:---:|:---:|:---:|:---:|
| **"I"** | -0.006 | 0.001 | **0.017** | 0.006 |
| **"can"** | 0.003 | -0.002 | 0.008 | 0.004 |
| **"go"** | 0.001 | 0.005 | 0.003 | **0.012** |
| **"alone"** | -0.003 | 0.002 | 0.004 | 0.007 |

These are the **logits** - raw prediction scores before converting to probabilities.

---

## Phase 6: Softmax and Loss

### Goal
Measure how wrong the model is.

### Files
- `src/Loss.hpp`
- `src/Loss.cpp`

### The Problem
The model outputs raw scores (logits). We need to:
1. Convert scores into probabilities (softmax)
2. Compare with the correct answer (cross-entropy loss)

### Part 1: Softmax

Softmax converts any list of numbers into probabilities that sum to 1.

**Formula:**
```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

**Analogy:** Imagine 4 people bidding on an item. Softmax turns their bids into probabilities of winning.

**Example:**
```
Logits:     [-0.006,  0.001,  0.017,  0.006]
              |         |        |        |
            exp()     exp()    exp()    exp()
              |         |        |        |
           0.994     1.001    1.017    1.006

Sum = 0.994 + 1.001 + 1.017 + 1.006 = 4.018

Probabilities:
  "I":     0.994 / 4.018 = 0.247
  "can":   1.001 / 4.018 = 0.249
  "go":    1.017 / 4.018 = 0.253  <- highest
  "alone": 1.006 / 4.018 = 0.250

Sum = 0.247 + 0.249 + 0.253 + 0.250 = 0.999 ~ 1.0
```

After softmax, "go" has probability 0.253 (25.3% chance).

### Part 2: Cross-Entropy Loss

Cross-entropy measures the difference between predicted probabilities and the true answer.

**Formula:**
```
loss = -log(probability_of_correct_answer)
```

**Example:**

For "I can go alone", the correct next words are:
- After "I" -> "can" (but model predicted "go")
- After "can" -> "go" (model predicted "go" - correct!)
- After "go" -> "alone" (model predicted "alone" - correct!)

**Loss for position 0 ("I" -> should be "can"):**
```
Model predicted: ["I": 0.247, "can": 0.249, "go": 0.253, "alone": 0.250]
Correct answer: "can" (probability = 0.249)

loss = -log(0.249) = -(-1.389) = 1.389
```

**Loss for position 1 ("can" -> should be "go"):**
```
Correct answer: "go" (probability = let's say 0.30)
loss = -log(0.30) = 1.204
```

**Total loss = average of all positions**

### Random Baseline

If the model guesses randomly among 257 tokens, the expected loss is:
```
loss = log(257) = 5.549
```

This is our starting point. A good model will have loss much lower than this.

---

## Phase 7: Backpropagation and Optimizer

### Goal
Make the model learn by updating weights.

### Files
- `src/Trainer.hpp` and `src/Trainer.cpp`
- Updates to: `Loss`, `Linear`, `Embedding`, `MultiHeadAttention`, `LayerNorm`, `FeedForward`

### The Problem
We know the model is wrong (high loss). Now we need to figure out:
1. Which weights caused the error?
2. How should we change them to reduce the error?

### The Solution: Chain Rule from Calculus

Backpropagation is like playing the **blame game** backward through the network.

**Analogy:** Imagine a factory assembly line:
- Station A (Embedding) makes parts
- Station B (Linear) assembles them
- Station C (Loss) checks quality

If the final product is bad:
1. Loss tells Linear: "Your output was wrong by this much"
2. Linear tells Embedding: "Your input caused this much of the problem"
3. Each station updates its own tools (weights)

### The Training Loop

```
For each batch:
  1. ZERO GRADIENTS
     - Erase all correction notes (gradients)

  2. FORWARD PASS
      - Input -> Embedding -> TransformerBlocks -> Linear -> Logits -> Loss

  3. COMPUTE LOSS
      - How wrong were the predictions?

  4. BACKWARD PASS
      - Loss computes grad_logits
      - Linear computes grad_weights, grad_bias, and grad_input
      - TransformerBlocks compute gradients through Attention + FeedForward + LayerNorm
      - Embedding computes grad_token_embeddings, grad_position_embeddings

  5. UPDATE WEIGHTS (SGD with Gradient Clipping)
      - Clip gradient L2 norm to max_norm = 1.0 per layer
      - new_weight = old_weight - learning_rate * clipped_gradient

  6. REPEAT
```

### Concrete Example: Backward Pass

**Scenario:** After "I", the model should predict "can", but it predicted "go".

**Blame from Loss layer (grad_output):**
```
["I": 0.2, "can": -0.6, "go": 0.3, "alone": 0.1]
```

Negative blame for "can" = we predicted too low, need to increase.
Positive blame for "I" = we predicted too high, need to decrease.

#### Linear Layer Backward

**Compute grad_bias:**
```
grad_bias["I"] += 0.2
grad_bias["can"] += -0.6
grad_bias["go"] += 0.3
grad_bias["alone"] += 0.1
```

**Compute grad_weights:**
```
grad_weights[feature_0]["can"] += input[0] * blame["can"]
                                 += 0.5 * (-0.6)
                                 += -0.3
```

Negative gradient = increasing this weight reduces loss = **increase it!**

**Compute grad_input (to pass to Embedding):**
```
grad_input[0] += blame["I"] * weights[0]["I"]
               + blame["can"] * weights[0]["can"]
               + blame["go"] * weights[0]["go"]
               + blame["alone"] * weights[0]["alone"]

grad_input[0] += (0.2 * 0.01) + (-0.6 * -0.02) + (0.3 * 0.03) + (0.1 * -0.01)
              += 0.002 + 0.012 + 0.009 - 0.001
              = 0.022
```

This tells Embedding: "Feature 0 of your output was slightly too low."

#### Embedding Layer Backward

The Embedding receives `grad_input` and accumulates it:
```
# For token "I" (token ID 0) at position 0:
grad_token_embeddings[0][0] += 0.022
grad_position_embeddings[0][0] += 0.022
```

#### Update Weights (Optimizer)

```cpp
# learning_rate = 0.01

# Update Linear weight
weights[0]["can"] = -0.02 - 0.01 * (-0.3)
                  = -0.02 + 0.003
                  = -0.017

# Update Linear bias
bias["can"] = 0.0 - 0.01 * (-0.6)
            = 0.0 + 0.006
            = 0.006

# Update Embedding
embedding["I"][0] = 0.5 - 0.01 * 0.022
                  = 0.5 - 0.00022
                  = 0.49978
```

After the update, the model will predict "can" slightly more strongly next time it sees "I".

### Training Results

After running 2000 training steps:
```
Step 1:     Loss = 5.549  (random baseline)
Step 500:   Loss = 5.272
Step 1000:  Loss = 5.022
Step 1500:  Loss = 4.802
Step 2000:  Loss = 4.590
```

The loss decreased from 5.55 to 4.59, proving the model is learning!

---

## Phase 8: Simple Text Generation

### Goal
Generate Darija-like text from the trained model.

### Files
- `src/Sampler.hpp`
- `src/Sampler.cpp`

### The Problem
After training, we want the model to create new text, not just predict the next token for existing text. How do we turn prediction scores into actual text?

### The Solution: Autoregressive Generation

The model predicts one token at a time. We use this to generate text character by character:

```
1. Start with a prompt (e.g., "salam")
2. Encode prompt to token IDs
3. Forward pass through model -> get prediction scores for last token
4. Convert scores to probabilities (softmax)
5. Sample one token from the probabilities
6. Append sampled token to the sequence
7. Go back to step 3 (now with the longer sequence)
8. Stop when EOT token or max length reached
```

**This is called autoregressive generation** because the model predicts the next token based on all previously generated tokens.

### Sampling Strategies

#### 1. Greedy Decoding (Temperature = 0)
Always pick the token with the highest probability.

```
Logits:     [0.5,  2.0,  1.0, -0.5]
Softmax:    [0.10, 0.60, 0.22, 0.08]
               ^     ^     ^     ^
              10%   60%   22%    8%

Greedy picks token 1 (60% probability)
```

**Pros:** Deterministic, always picks the "safest" choice.
**Cons:** Boring, repetitive text (tends to get stuck in loops).

#### 2. Temperature Sampling (Temperature > 0)
Divide logits by temperature before softmax. This controls randomness:

- **Temperature = 0.1** (low): Almost greedy, slight randomness
- **Temperature = 0.8** (medium): Balanced creativity
- **Temperature = 1.5** (high): Very random, creative but often nonsensical

**How it works:**

```
Original logits:     [2.0, 1.0, 0.5]

Temperature = 0.5 (focused):
  Scaled:            [4.0, 2.0, 1.0]
  Softmax:           [0.84, 0.12, 0.04]
  ^ High confidence, one clear winner

Temperature = 2.0 (random):
  Scaled:            [1.0, 0.5, 0.25]
  Softmax:           [0.52, 0.31, 0.17]
  ^ More uniform, more random choices
```

**Analogy:** Temperature is like a creative dial:
- Low temp = cautious writer who always picks the most common word
- High temp = creative writer who takes risks and surprises you

### Concrete Example

**Prompt:** `"salam "`

**Generation with greedy (temp=0):**
```
Step 1: "salam " -> predict 'a' (highest score)
Step 2: "salam a" -> predict 'a' (highest score)
Step 3: "salam aa" -> predict 'a' (highest score)
...
Result: "salam aaaaaaaaaaaaaaaaa..." (gets stuck!)
```

**Generation with temperature=0.8:**
```
Step 1: "salam " -> predict '3' (prob=0.15)
Step 2: "salam 3" -> predict 'l' (prob=0.12)
Step 3: "salam 3l" -> predict 'i' (prob=0.18)
...
Result: "salam 3likom kayn ..." (varied, sometimes Darija-like)
```

### Why the Output is Messy

Our model is very small:
- Only 128 embedding dimensions (small but workable)
- No transformer blocks yet (just one attention layer)
- Trained for only 10,000 steps (configurable)

With a larger model (Phase 14), the output becomes much cleaner and more coherent.

---

## Phase 9: Self-Attention

### Goal
Let each token look at previous tokens to understand context.

### Files
- `src/MultiHeadAttention.hpp`
- `src/MultiHeadAttention.cpp`

> Note: This project implements multi-head attention directly. The single-head `Attention` files mentioned in earlier educational drafts were merged into `MultiHeadAttention`.

### The Problem
Until now, each token only knew about itself. The model predicted the next token based on the current token's embedding, but it couldn't look at what came before.

For example, in the sentence "I saw a cat, it was cute", the word "it" refers to "cat". The model needs to look back at previous words to understand this relationship.

### The Solution: Self-Attention

**Self-attention** allows each token to "look at" all previous tokens and decide which ones are important.

### The Attention Formula

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Where:
- **Q** (Query): "What am I looking for?"
- **K** (Key): "What do I contain?"
- **V** (Value): "What information do I have?"

### How It Works Step by Step

#### Step 1: Create Q, K, V projections

For each token's embedding vector, we compute three new vectors by multiplying with weight matrices:

```
Q = X @ Wq    (Query: what this token is looking for)
K = X @ Wk    (Key: what this token contains)
V = X @ Wv    (Value: information this token has)
```

**Concrete Example:**

Input embeddings (2 tokens, 4 dimensions each):
```
Token 0 "I":    [0.5, 0.1, -0.2, 0.3]
Token 1 "can":  [0.2, 0.4, 0.1, -0.1]
```

Weight matrices (4x4 each):
```
Wq = [[0.01, -0.02, 0.03, -0.01], ...]  (4x4)
Wk = [[0.02, 0.01, -0.01, 0.03], ...]   (4x4)
Wv = [[-0.01, 0.02, 0.01, -0.02], ...]  (4x4)
```

Compute Q[0] = X[0] @ Wq:
```
Q[0][0] = 0.5*0.01 + 0.1*(-0.02) + (-0.2)*0.03 + 0.3*(-0.01) = -0.008
Q[0][1] = 0.5*(-0.02) + 0.1*0.01 + (-0.2)*(-0.01) + 0.3*0.03 = 0.008
...
Q[0] = [-0.008, 0.008, ...]
```

#### Step 2: Compute Attention Scores

For each pair of tokens (i, j), compute how much token i should attend to token j:

```
scores[i][j] = Q[i] dot K[j] / sqrt(embed_dim)
```

**Example with 2 tokens:**
```
scores[0][0] = Q[0] dot K[0] / 2.0 = 0.15
scores[0][1] = Q[0] dot K[1] / 2.0 = 0.08
scores[1][0] = Q[1] dot K[0] / 2.0 = 0.12
scores[1][1] = Q[1] dot K[1] / 2.0 = 0.20
```

#### Step 3: Apply Causal Mask

In language models, we only allow looking at **previous** tokens, not future ones. We mask out future positions by setting their scores to -infinity:

```
scores[i][j] = -inf  if j > i
```

**Masked scores table:**
```
        Token 0   Token 1
Token 0   0.15     -inf     <- "I" can only see itself
Token 1   0.12      0.20    <- "can" can see "I" and itself
```

**Why causal mask?** When predicting token 1, the model shouldn't be allowed to "cheat" by looking at token 2. It can only use what it has already seen.

#### Step 4: Softmax

Apply softmax to each row to get attention weights (probabilities that sum to 1):

```
Row 0: [0.15, -inf] -> softmax -> [1.0, 0.0]
Row 1: [0.12, 0.20] -> softmax -> [0.45, 0.55]
```

**Attention Weights Table:**
```
         Token 0   Token 1
Token 0   1.00     0.00
Token 1   0.45     0.55
```

**Interpretation:**
- When processing "I" (position 0), it pays 100% attention to itself.
- When processing "can" (position 1), it pays 45% attention to "I" and 55% to itself.

#### Step 5: Weighted Sum of Values

For each position, compute the output as a weighted sum of all Value vectors:

```
output[i] = sum_j (weights[i][j] * V[j])
```

**Example:**
```
output[0] = 1.00 * V[0] + 0.00 * V[1] = V[0]
output[1] = 0.45 * V[0] + 0.55 * V[1]
```

The output for position 1 is a mix of information from "I" and "can", weighted by how relevant each is.

### Visual Diagram: Self-Attention Flow

```
Input Embeddings (X)
    |
    |    Wq      Wk      Wv
    |     |       |       |
    v     v       v       v
   [Q]   [K]     [V]
    |     |       |
    |_____|       |
    |  dot        |
    v             |
  scores          |
    |             |
  causal mask     |
    |             |
  softmax         |
    |             |
  weights         |
    |_____________|
          |
          v
      output
```

### The Complete Attention Matrix

For a sentence with 4 words, the attention matrix shows how much each word looks at every other word:

```
        "I"   "can"   "go"   "alone"
"I"      1.0    0.0    0.0     0.0
"can"    0.3    0.7    0.0     0.0
"go"     0.1    0.2    0.7     0.0
"alone"  0.05   0.05   0.1     0.8
```

**Diagonal is always strongest:** Each word pays most attention to itself.
**Lower triangle only:** Due to causal mask, words cannot see the future.
**Earlier words get less attention:** As the sentence grows, older words become less relevant.

### Why Self-Attention is Powerful

1. **Long-range dependencies:** The word "it" can directly attend to "cat" from 10 words ago.
2. **Parallel computation:** All attention scores are computed simultaneously.
3. **Interpretable:** We can visualize the attention matrix to see what the model is "thinking."

### Training with Attention

The updated training pipeline now includes the attention layer:

```
FORWARD:
  Input tokens -> Embedding -> Attention -> FeedForward -> Linear -> Logits -> Loss

BACKWARD:
  Loss -> Linear -> FeedForward -> Attention -> Embedding
```

The attention layer has three sets of weights (Wq, Wk, Wv) that are learned during training.

### Training Results with Attention

After adding attention and training 10,000 steps:
```
Step 1:     Loss = 5.549  (random baseline)
Step 1000:  Loss = 4.991
Step 2000:  Loss = 4.542
```

The loss is similar to before (4.54 vs 4.59) because our model is still very small. But attention lays the foundation for the much more powerful model in Phase 14.

---

## Phase 10: Multi-Head Attention

### Goal
Run multiple attention heads in parallel, then combine their results.

### Files
- `src/MultiHeadAttention.hpp`
- `src/MultiHeadAttention.cpp`

### The Problem
A single attention head can only focus on one type of relationship at a time. For example, in the sentence "I saw a cat, it was cute":
- One relationship: "it" refers to "cat" (pronoun -> noun)
- Another relationship: "saw" is a verb related to "cat" (verb -> object)
- Another relationship: "cute" describes "cat" (adjective -> noun)

A single head must try to capture all these patterns at once. **Multi-head attention** gives the model multiple "slots" to learn different types of relationships.

### The Solution: Multiple Heads

Instead of one set of Q, K, V projections, we create **num_heads** sets.

**Analogy:** Imagine a team of detectives investigating a case:
- Detective 1 focuses on "who did what"
- Detective 2 focuses on "where things happened"
- Detective 3 focuses on "when things happened"
- Each detective looks at the same evidence but from a different angle
- At the end, they combine their findings into one report

### How It Works

#### Step 1: Split Embedding Dimension into Heads

Given:
- `embed_dim = 128`
- `num_heads = 4`
- `head_dim = embed_dim / num_heads = 32`

Each head gets its own slice of 8 dimensions:

```
Full embedding vector (128 dims):
[ h0_d0, h0_d1, h0_d2, h0_d3, h0_d4, h0_d5, h0_d6, h0_d7 |  <- Head 0 (8 dims)
  h1_d0, h1_d1, h1_d2, h1_d3, h1_d4, h1_d5, h1_d6, h1_d7 |  <- Head 1 (8 dims)
  h2_d0, h2_d1, h2_d2, h2_d3, h2_d4, h2_d5, h2_d6, h2_d7 |  <- Head 2 (8 dims)
  h3_d0, h3_d1, h3_d2, h3_d3, h3_d4, h3_d5, h3_d6, h3_d7 ]  <- Head 3 (8 dims)
```

#### Step 2: Compute Q, K, V for All Heads

We use **combined projection matrices** (embed_dim x embed_dim):
- `Wq` = [Wq_head0 | Wq_head1 | Wq_head2 | Wq_head3]
- `Wk` = [Wk_head0 | Wk_head1 | Wk_head2 | Wk_head3]
- `Wv` = [Wv_head0 | Wv_head1 | Wv_head2 | Wv_head3]

Each column slice (embed_dim x head_dim) belongs to one head.

```
Q = X @ Wq    -> shape (batch, seq, 32)
K = X @ Wk    -> shape (batch, seq, 32)
V = X @ Wv    -> shape (batch, seq, 32)
```

#### Step 3: Run Attention Independently Per Head

For each head h (0 to 3):

```
Q_h = Q[:, :, h*8 : (h+1)*8]     -> shape (batch, seq, 8)
K_h = K[:, :, h*8 : (h+1)*8]     -> shape (batch, seq, 8)
V_h = V[:, :, h*8 : (h+1)*8]     -> shape (batch, seq, 8)

scores_h = Q_h @ K_h^T / sqrt(8)   -> shape (seq, seq)
weights_h = softmax(causal_mask(scores_h))  -> shape (seq, seq)
output_h = weights_h @ V_h         -> shape (seq, 8)
```

**Concrete Example with 2 Heads:**

Input: "I can" (2 tokens, embed_dim = 4, num_heads = 2, head_dim = 2)

**Head 0:** Focuses on subject-verb relationships
```
Attention weights (Head 0):
         "I"    "can"
"I"      1.00    0.00
"can"    0.70    0.30
```
- "can" pays 70% attention to "I" (subject -> verb link)

**Head 1:** Focuses on position/tense
```
Attention weights (Head 1):
         "I"    "can"
"I"      1.00    0.00
"can"    0.20    0.80
```
- "can" pays 80% attention to itself (modal verb focuses on its own features)

#### Step 4: Concatenate Head Outputs

```
concat = [output_0 | output_1 | output_2 | output_3]
        -> shape (batch, seq, 32)
```

**Concatenation visual:**
```
Token "can":
  Head 0 output: [0.1, -0.2, 0.3, 0.0, 0.1, -0.1, 0.2, 0.0]
  Head 1 output: [0.0, 0.1, -0.1, 0.2, 0.0, 0.1, -0.2, 0.1]
  Head 2 output: [0.2, 0.0, 0.1, -0.1, 0.0, 0.2, 0.1, -0.1]
  Head 3 output: [-0.1, 0.1, 0.0, 0.1, 0.2, -0.1, 0.0, 0.2]
  -----------------------------------------------------------
  Concatenated:  [0.1, -0.2, 0.3, 0.0, 0.1, -0.1, 0.2, 0.0,
                   0.0, 0.1, -0.1, 0.2, 0.0, 0.1, -0.2, 0.1,
                   0.2, 0.0, 0.1, -0.1, 0.0, 0.2, 0.1, -0.1,
                  -0.1, 0.1, 0.0, 0.1, 0.2, -0.1, 0.0, 0.2]
```

#### Step 5: Output Projection (Wo)

The concatenated output is mixed together using an output projection matrix:

```
final_output = concat @ Wo     -> shape (batch, seq, 32)
```

**Why Wo?** Different heads might produce redundant or conflicting information. Wo learns how to blend them optimally.

### Visual Diagram: Multi-Head Attention Flow

```
Input Embeddings (X)
    |
    |    Wq (32x32)      Wk (32x32)      Wv (32x32)
    |     |                |                |
    v     v                v                v
   [Q]   [K]              [V]
    |     |                |
    |  Split into 4 heads  |
    |     |                |
    |   Head 0  Head 1  Head 2  Head 3
    |     |       |       |       |
    |   Q0,K0,V0 Q1,K1,V1 Q2,K2,V2 Q3,K3,V3
    |     |       |       |       |
    |   Attn0   Attn1   Attn2   Attn3
    |     |       |       |       |
    |  out_0   out_1   out_2   out_3
    |     |       |       |       |
    |     +-------+-------+-------+
    |             |
    |          Concatenate
    |             |
    |          (128 dims)
    |             |
    |          Wo (32x32)
    |             |
    v             v
            final_output
```

### Shape Summary

| Tensor | Shape | Description |
|---|---|---|
| X (input) | (batch, seq, 32) | Input embeddings |
| Wq, Wk, Wv | (32, 32) | Combined projection matrices |
| Q, K, V | (batch, seq, 32) | Projected embeddings (all heads) |
| Q_h, K_h, V_h | (batch, seq, 8) | Per-head tensors (h = 0..3) |
| scores_h | (seq, seq) | Attention scores for head h |
| weights_h | (seq, seq) | Softmax weights for head h |
| concat | (batch, seq, 32) | Concatenated head outputs |
| Wo | (32, 32) | Output projection matrix |
| output | (batch, seq, 32) | Final multi-head attention output |

### Backward Pass

The backward pass is an extension of single-head backward:

1. **dL/dWo**: `concat^T @ grad_output`
2. **dL/dconcat**: `grad_output @ Wo^T`
3. **Per-head backward**: For each head, run the single-head attention backward to get `grad_Q_h`, `grad_K_h`, `grad_V_h`
4. **Reassemble**: Place each head's gradients back into full `grad_Q`, `grad_K`, `grad_V`
5. **dL/dWq, dL/dWk, dL/dWv**: `X^T @ grad_Q`, `X^T @ grad_K`, `X^T @ grad_V`
6. **dL/dX**: `grad_Q @ Wq^T + grad_K @ Wk^T + grad_V @ Wv^T`

### Training Results with Multi-Head Attention

After adding 4-head attention and training 10,000 steps:
```
Step 1:     Loss = 5.549  (random baseline)
Step 1000:  Loss = 4.989
Step 2000:  Loss = 4.627
```

The loss reaches ~4.56, similar to single-head attention. The benefit of multi-head attention becomes clear with deeper models (Phase 14), where different heads learn specialized linguistic patterns.

### Why Multiple Heads Help

1. **Specialization:** Each head can learn a different type of relationship
   - Head 0: subject-verb agreement
   - Head 1: pronoun references
   - Head 2: local word patterns
   - Head 3: long-range dependencies

2. **Redundancy:** If one head fails to learn a pattern, another head might succeed

3. **Richer representations:** Concatenating multiple perspectives creates a more expressive output than any single head could produce

---

## Phase 11: Feed-Forward Network

### Goal
Add a small neural network after attention to let each token "think" independently about what it learned from other tokens.

### Files
- `src/FeedForward.hpp`
- `src/FeedForward.cpp`

### The Problem
After attention mixes information between tokens, each token still needs its own transformation layer. Attention tells tokens what other tokens are saying, but the feed-forward network lets each token decide what to do with that information.

**Analogy:** In a meeting:
- **Attention** = everyone shares their ideas (information mixing)
- **Feed-Forward** = each person privately thinks about what they heard and forms their own conclusion

### Architecture

The feed-forward network has a simple structure:

```
Input (embed_dim)
    |
    v
Linear Up: embed_dim -> 4 * embed_dim
    |
    v
GELU Activation
    |
    v
Linear Down: 4 * embed_dim -> embed_dim
    |
    v
Output (embed_dim)
```

**Why 4x expansion?**
We first expand the dimension to give the model more "thinking space," then compress back. This lets the model learn richer transformations than it could at the original size.

### Concrete Example

**Setup:**
- `embed_dim = 4`
- Hidden dimension = `4 * 4 = 16`

**Input (one token after attention):**
```
X = [0.5, -0.2, 0.3, 0.1]
```

**Step 1: Linear Up (4 -> 16)**

Weight matrix W_up (4 rows x 16 columns):
```
         out0  out1  out2  ...  out15
feat 0:  0.01 -0.02  0.03       0.01
feat 1: -0.01  0.02 -0.01       0.03
feat 2:  0.02 -0.01  0.02      -0.02
feat 3: -0.02  0.03  0.01       0.01
```

```
hidden[0] = 0.5*0.01 + (-0.2)*(-0.01) + 0.3*0.02 + 0.1*(-0.02)
          = 0.005 + 0.002 + 0.006 - 0.002
          = 0.011

hidden[1] = 0.5*(-0.02) + (-0.2)*0.02 + 0.3*(-0.01) + 0.1*0.03
          = -0.01 - 0.004 - 0.003 + 0.003
          = -0.014

... (repeat for all 16 outputs)
```

**Result:** `hidden = [0.011, -0.014, ..., 0.008]` (16 numbers)

**Step 2: GELU Activation**

GELU is a smooth version of ReLU. It keeps positive values mostly unchanged but smoothly suppresses negative values.

**Formula:**
```
gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

Where `erf` is the error function (a sigmoid-like curve).

**Visual comparison:**
```
     ReLU:              GELU:
       |                   |
    2  |              2    |     .----
       |  /                |    /
    1  | /             1   |   /
       |/                 |  /
    0  +------>        0  +-+------>
       |                 |/|
   -1  |            -1   / |
       |               /   |
   -2  |           -2  /   |
```

ReLU is sharp at zero. GELU is smooth everywhere, which helps gradients flow better during training.

**Applying GELU to our hidden vector:**
```
hidden[0] = 0.011   -> gelu(0.011)   = 0.006
hidden[1] = -0.014  -> gelu(-0.014)  = -0.007
... (small negative values stay small negative)
```

**Step 3: Linear Down (16 -> 4)**

Weight matrix W_down (16 rows x 4 columns):
```
         out0  out1  out2  out3
hid 0:   0.02 -0.01  0.03 -0.02
hid 1:  -0.01  0.02 -0.01  0.03
... (16 rows)
```

```
output[0] = 0.006*0.02 + (-0.007)*(-0.01) + ...
          = 0.00012 + 0.00007 + ...
          = 0.003

output[1] = 0.006*(-0.01) + (-0.007)*0.02 + ...
          = -0.00006 - 0.00014 + ...
          = -0.001

... (repeat for all 4 outputs)
```

**Final output:** `[0.003, -0.001, 0.002, 0.004]` (4 numbers, same shape as input)

### Shape Summary

| Tensor | Shape | Description |
|---|---|---|
| Input | (batch, seq, embed_dim) | Token vectors after attention |
| After Linear Up | (batch, seq, 4*embed_dim) | Expanded representations |
| After GELU | (batch, seq, 4*embed_dim) | Activated hidden states |
| After Linear Down | (batch, seq, embed_dim) | Final output (same shape as input) |

### Backward Pass

The backward pass goes through each component in reverse:

```
grad_output (from next layer)
    |
    v
Linear Down backward:
  - grad_W_down = activated_hidden^T @ grad_output
  - grad_activated = grad_output @ W_down^T
    |
    v
GELU backward:
  - grad_hidden = grad_activated * gelu_derivative(hidden)
    |
    v
Linear Up backward:
  - grad_W_up = input^T @ grad_hidden
  - grad_input = grad_hidden @ W_up^T (passed to previous layer)
```

**GELU derivative:**
```
gelu'(x) = 0.5 * (1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi)
         = phi + x * pdf
```

Where `phi` is the cumulative normal distribution and `pdf` is the normal probability density. This derivative tells us how much to change the hidden state to reduce the loss.

### Visual Diagram: Feed-Forward Network

```
Input Embeddings (X)
    |
    |    W_up (E x 4E)
    |     |
    v     v
   [Hidden]
    |
    |  GELU
    v
[Activated]
    |
    |    W_down (4E x E)
    |     |
    v     v
   [Output]
```

### Why Feed-Forward Matters

1. **Independent processing:** Each token transforms its own vector without looking at neighbors (that was attention's job)
2. **Non-linearity:** GELU introduces non-linear transformations, allowing the model to learn complex patterns
3. **Capacity expansion:** The 4x hidden dimension gives the model room to learn rich intermediate representations
4. **Same input/output shape:** Can be stacked repeatedly without changing tensor dimensions

---

## Phase 12: LayerNorm and TransformerBlock

### Goal
Make training stable with normalization and skip connections.

### Files
- `src/LayerNorm.hpp`
- `src/LayerNorm.cpp`
- `src/TransformerBlock.hpp`
- `src/TransformerBlock.cpp`

### The Problem
Deep neural networks suffer from two major issues during training:
1. **Vanishing gradients:** Gradients get smaller and smaller as they propagate backward through many layers
2. **Exploding activations:** As data flows forward through many layers, values can grow exponentially large or shrink to near zero

These problems make deep networks nearly impossible to train without special techniques.

### Solution 1: Layer Normalization

LayerNorm normalizes the values within each token's embedding vector, keeping them in a healthy range.

**Formula:**
```
y = (x - mean) / sqrt(variance + epsilon) * gamma + beta
```

Where:
- `mean` = average of all values in the embedding vector
- `variance` = how spread out the values are
- `epsilon` = tiny number (1e-5) to prevent division by zero
- `gamma` = learnable scale parameter (initialized to 1.0)
- `beta` = learnable shift parameter (initialized to 0.0)

**Analogy:** Imagine a group of people reporting temperatures in different units (Celsius, Fahrenheit, Kelvin). Normalization converts them all to a standard scale (like z-scores) so they can be compared fairly.

### Concrete Example: LayerNorm

**Input vector (one token, embed_dim = 4):**
```
x = [1.0, 2.0, 3.0, 4.0]
```

**Step 1: Compute mean**
```
mean = (1.0 + 2.0 + 3.0 + 4.0) / 4 = 10.0 / 4 = 2.5
```

**Step 2: Compute variance**
```
variance = ((1.0-2.5)^2 + (2.0-2.5)^2 + (3.0-2.5)^2 + (4.0-2.5)^2) / 4
         = (2.25 + 0.25 + 0.25 + 2.25) / 4
         = 5.0 / 4
         = 1.25
```

**Step 3: Normalize**
```
std = sqrt(1.25 + 0.00001) = 1.118

x_hat[0] = (1.0 - 2.5) / 1.118 = -1.341
x_hat[1] = (2.0 - 2.5) / 1.118 = -0.447
x_hat[2] = (3.0 - 2.5) / 1.118 =  0.447
x_hat[3] = (4.0 - 2.5) / 1.118 =  1.341
```

**Step 4: Apply learnable gamma and beta**
```
# gamma = [1.0, 1.0, 1.0, 1.0] (initial)
# beta  = [0.0, 0.0, 0.0, 0.0] (initial)

y[0] = -1.341 * 1.0 + 0.0 = -1.341
y[1] = -0.447 * 1.0 + 0.0 = -0.447
y[2] =  0.447 * 1.0 + 0.0 =  0.447
y[3] =  1.341 * 1.0 + 0.0 =  1.341
```

**Result:** The values are now centered around 0 with standard deviation ~1, regardless of what the original scale was.

**Visual of LayerNorm effect:**
```
Before LayerNorm:          After LayerNorm:
[1.0, 2.0, 3.0, 4.0]      [-1.34, -0.45, 0.45, 1.34]
  |    |    |    |          |      |     |     |
  Large spread               Standardized spread
  Mean = 2.5                 Mean = 0
  Std = 1.12                 Std = 1
```

### Solution 2: Residual Connections (Skip Connections)

Residual connections add the input directly to the output of a layer:
```
output = input + sublayer(input)
```

**Analogy:** Imagine renovating a house:
- Without residual: you completely replace the old structure with a new one (information from early layers gets lost)
- With residual: you keep the old structure and add improvements on top (original information always flows through)

**Why residuals work:**
1. **Gradient highway:** Gradients can flow directly through the `+` operation without passing through layers, preventing vanishing gradients
2. **Identity path:** The network only needs to learn the "delta" or "improvement" to the input, not a completely new representation
3. **Deeper networks:** Allows stacking many layers (100+ in GPT-3) while maintaining stable training

### Concrete Example: Residual Connection

**Input:** `x = [0.5, -0.2, 0.3, 0.1]`

**After Attention sublayer:** `attention_out = [0.3, 0.1, -0.1, 0.2]`

**With residual:**
```
output = x + attention_out
       = [0.5, -0.2, 0.3, 0.1] + [0.3, 0.1, -0.1, 0.2]
       = [0.8, -0.1, 0.2, 0.3]
```

The output preserves the original input while adding the attention transformation on top.

### The Full Transformer Block

A Transformer block combines LayerNorm, Attention, FeedForward, and Residuals into one reusable unit:

```
Input (batch, seq, embed_dim)
    |
    v
+-----------+
| LayerNorm |  <- Normalize input
+-----------+
    |
    v
+-------------------+
| MultiHeadAttention|  <- Mix information between tokens
+-------------------+
    |
    v
+--------+   Input (residual connection)
|   +    |<--/
+--------+
    |
    v
+-----------+
| LayerNorm |  <- Normalize again
+-----------+
    |
    v
+-----------------+
| FeedForward     |  <- Independent token processing
| (up -> GELU ->  |
|      down)      |
+-----------------+
    |
    v
+--------+   Previous output (residual connection)
|   +    |<--/
+--------+
    |
    v
Output (batch, seq, embed_dim)
```

**Visual flow with actual data (one token):**
```
Input x:              [0.5, -0.2, 0.3, 0.1]
                          |
                          v
After LN1:            [0.8, -1.2, 0.4, 0.0]   (normalized)
                          |
                          v
After Attention:      [0.3, 0.1, -0.1, 0.2]
                          |
                          v
Residual Add:         [0.8, -0.1, 0.2, 0.3]   = x + attention
                          |
                          v
After LN2:            [1.2, -0.8, 0.0, 0.6]   (normalized)
                          |
                          v
After FeedForward:    [0.1, 0.2, -0.1, 0.0]
                          |
                          v
Residual Add:         [0.9, 0.1, 0.1, 0.3]   = previous + ff
                          |
                          v
Final Output:         [0.9, 0.1, 0.1, 0.3]   (same shape as input!)
```

### Why This Order Matters

The standard Transformer uses **Pre-Norm** (LayerNorm before each sublayer):
```
x = x + Attention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

**Benefits of Pre-Norm:**
1. **Stable gradients:** Normalizing before the sublayer prevents large values from entering attention/FFN
2. **Easier training:** Works better for very deep networks (our model is shallow but this scales well)
3. **Standard approach:** Used in GPT, BERT, and most modern transformers

**Alternative (Post-Norm):** LayerNorm after the sublayer + residual. This was used in the original Transformer paper but is harder to train.

### Shape Summary

| Tensor | Shape | Description |
|---|---|---|
| Input | (batch, seq, embed_dim) | Embeddings from previous layer |
| After LN1 | (batch, seq, embed_dim) | Normalized input |
| After Attention | (batch, seq, embed_dim) | Mixed token information |
| After First Residual | (batch, seq, embed_dim) | Input + Attention output |
| After LN2 | (batch, seq, embed_dim) | Normalized again |
| After FeedForward | (batch, seq, embed_dim) | Transformed representations |
| After Second Residual | (batch, seq, embed_dim) | Previous + FeedForward output |
| Final Output | (batch, seq, embed_dim) | Same shape as input! |

### Backward Pass Through TransformerBlock

Backpropagation through residuals is elegant because gradients flow through both paths:

```
Gradients from next layer (grad_output)
    |
    +-----> Direct path (residual)
    |       grad_after_ff = grad_output
    |
    +-----> Through FeedForward
            grad_ln2 = FeedForward.backward(grad_output)
            |
            +-----> Direct path (residual)
            |       grad_after_attn = grad_ln2
            |
            +-----> Through LayerNorm2
                    grad_ln2_out = LayerNorm2.backward(grad_ln2)
                    |
                    +-----> Through Attention
                            grad_ln1 = Attention.backward(grad_ln2_out)
                            |
                            +-----> Direct path (residual)
                            |       grad_input = grad_ln1
                            |
                            +-----> Through LayerNorm1
                                    grad_final = LayerNorm1.backward(grad_ln1)
                                    |
                                    v
                            Final gradient to previous layer
```

The gradient gets **added** at each residual branch, creating multiple paths for gradients to flow backward. This prevents vanishing gradients because even if one path becomes weak, the direct skip path remains strong.

### Visual Diagram: Gradient Flow

```
Loss
  |
  v
dL/doutput
  |
  +-----> [+]<---- dL/doutput (skip path)
  |         |
  |         v
  |    FeedForward.backward()
  |         |
  |         v
  |     dL/dLN2
  |         |
  |    +-->[+]<---- dL/dLN2 (skip path)
  |    |    |
  |    |    v
  |    | LayerNorm2.backward()
  |    |    |
  |    |    v
  |    | dL/dAttention
  |    |    |
  |    |    v
  |    | Attention.backward()
  |    |    |
  |    |    v
  |    | dL/dLN1
  |    |    |
  |    +-->[+]<---- dL/dLN1 (skip path)
  |         |
  |         v
  |    LayerNorm1.backward()
  |         |
  |         v
  +-------->+
            |
            v
      Final dL/dInput
```

### Why LayerNorm + Residuals Are Essential

**Without these techniques:**
- Training a 2-layer network becomes unstable
- Loss oscillates wildly or gets stuck
- Gradients vanish in early layers

**With these techniques:**
- Can train networks with 12, 24, or even 96 layers
- Loss decreases smoothly
- Gradients flow healthily to all layers

This combination is what made Transformers scalable and led to GPT, BERT, and other large language models.

### Training Pipeline (Updated)

With all components from Phases 1-12, the full training pipeline is:

```
FORWARD PASS:
  Input tokens
      |
      v
  Embedding (token + position)
      |
      v
  TransformerBlock:
    - LayerNorm1 -> MultiHeadAttention -> Residual
    - LayerNorm2 -> FeedForward -> Residual
      |
      v
  Linear projection
      |
      v
  Softmax -> Loss

BACKWARD PASS:
  Loss gradients
      |
      v
  Linear backward
      |
      v
  TransformerBlock backward:
    - FeedForward backward
    - LayerNorm2 backward
    - Attention backward
    - LayerNorm1 backward
      |
      v
  Embedding backward

UPDATE:
  All weights updated with SGD
```

---

## Phase 13: Transformer Block

### Goal
Combine all components into one reusable unit.

### Files
- `src/TransformerBlock.hpp`
- `src/TransformerBlock.cpp`

### What It Is
A TransformerBlock wires together:
1. LayerNorm
2. Multi-Head Attention
3. Residual connection
4. LayerNorm
5. Feed-Forward Network
6. Residual connection

### Architecture

```
input
  -> LayerNorm
  -> MultiHeadAttention
  -> + (residual)
  -> LayerNorm
  -> FeedForward
  -> + (residual)
  -> output
```

**Key property:** Input and output have the same shape `[batch, seq, embed_dim]`. This means you can stack many blocks one after another.

### Why Residuals Work

Without residual connections:
```
output = Attention(input)   <- original signal is lost
```

With residual connections:
```
output = input + Attention(input)   <- original signal is preserved
```

The network only needs to learn the **improvement** (delta) to the input, not a completely new representation. This makes training much easier.

### Why LayerNorm Works

Without normalization, values grow exponentially as they pass through layers:
```
Layer 1: [0.5, -0.2, 0.3]
Layer 2: [2.1, -0.8, 1.5]
Layer 3: [8.4, -3.2, 6.0]   <- exploding!
```

With LayerNorm, values stay in a healthy range:
```
Layer 1: [-0.5, 1.2, -0.7]
Layer 2: [0.3, -1.1, 0.8]
Layer 3: [-0.8, 0.4, 0.4]   <- stable!
```

### Concrete Example

Input to block: `[0.5, -0.2, 0.3, 0.1]`

```
After LayerNorm1:   [0.8, -1.2, 0.4, 0.0]   (normalized)
After Attention:    [0.3, 0.1, -0.1, 0.2]
After Residual:     [0.8, -0.1, 0.2, 0.3]   = input + attention
After LayerNorm2:   [1.2, -0.8, 0.0, 0.6]   (normalized)
After FeedForward:  [0.1, 0.2, -0.1, 0.0]
After Residual:     [0.9, 0.1, 0.1, 0.3]   = previous + ff
```

The output preserves the original signal while adding transformations on top.

---

## Phase 14: Mini GPT Model

### Goal
Build the complete GPT language model.

### Files
- `src/GPT.hpp`
- `src/GPT.cpp`

### Architecture

```
token IDs
  |
  v
Embedding (token + position)
  |
  v
TransformerBlock 1
  |
  v
TransformerBlock 2
  |
  v
... (more blocks)
  |
  v
Final LayerNorm
  |
  v
Linear Language Head
  |
  v
Logits over vocabulary
```

### Model Config

```text
vocab_size = 257      (bytes 0-255 + EOT)
context_length = 64   (max sequence length)
embed_dim = 64        (vector size per token)
num_heads = 2         (attention heads)
num_layers = 1        (transformer blocks)
```

### Why This Is Called GPT

**G**enerative **P**re-trained **T**ransformer:
- **Generative:** Predicts the next token
- **Pre-trained:** Trained on large text corpus first
- **Transformer:** Uses attention mechanisms

Our model is a tiny version of this architecture.

### Forward Pass

```cpp
auto logits = model.forward(batch_inputs);
// logits shape: [batch_size][context_length][vocab_size]
```

For each position in the sequence, the model outputs one score for every possible token in the vocabulary.

### Generate Text

```cpp
auto tokens = model.generate(prompt, max_tokens, temperature);
```

1. Encode prompt to token IDs
2. Forward pass to get logits for last token
3. Sample next token from probability distribution
4. Append token to sequence
5. Repeat until EOT or max length

---

## Phase 15: Training the Full Model

### Goal
Train the GPT model on Darija text.

### Files
- `src/Trainer.hpp`
- `src/Trainer.cpp`

### Training Loop

```
For each step:
  1. Get random batch from dataset
  2. model.zero_grad()
  3. logits = model.forward(inputs)
  4. loss = cross_entropy(logits, targets)
  5. grad_logits = loss.backward()
  6. model.backward(inputs, grad_logits)
  7. model.apply_gradients(learning_rate)
```

### Expected Loss

```text
Random baseline:    ~5.55 (log(257))
After 500 steps:    ~5.0
After 1000 steps:   ~4.5
After 10,000 steps:   ~3.5-4.0 (depending on dataset size)
```

Lower loss = better predictions. The model learns Darija patterns from the data.

### Evaluating

After training, we evaluate on both train and validation sets:
- **Train loss:** How well the model memorizes training data
- **Validation loss:** How well the model generalizes to unseen data

If validation loss is much higher than train loss, the model is **overfitting** (memorizing instead of learning patterns).

---

## Phase 16: Save and Load

### Goal
Save trained weights and load them later.

### Files
- `models/darija_gpt.bin`

### Binary Format

The model is saved as a binary file with this structure:

```
Header (28 bytes):
  Magic:    "DARIJGPT" (8 bytes)
  Version:  1 (4 bytes)
  Config:   vocab_size, context_length, embed_dim,
            num_heads, num_layers (5 x 4 bytes)

Weights:
  Embedding token embeddings     [vocab_size][embed_dim]
  Embedding position embeddings  [context_length][embed_dim]
  For each transformer block:
    LayerNorm1 gamma             [embed_dim]
    LayerNorm1 beta              [embed_dim]
    Attention Wq                 [embed_dim][embed_dim]
    Attention Wk                 [embed_dim][embed_dim]
    Attention Wv                 [embed_dim][embed_dim]
    Attention Wo                 [embed_dim][embed_dim]
    LayerNorm2 gamma             [embed_dim]
    LayerNorm2 beta              [embed_dim]
    FeedForward W_up             [embed_dim][4*embed_dim]
    FeedForward W_down           [4*embed_dim][embed_dim]
  Final LayerNorm gamma          [embed_dim]
  Final LayerNorm beta           [embed_dim]
  Language head weights          [embed_dim][vocab_size]
  Language head bias             [vocab_size]
```

### Why Save?

Training takes time. Saving lets you:
1. Train once, use many times
2. Resume training later
3. Share the model with others
4. Deploy to chat applications

### Loading

When loading, the program checks:
1. Magic header matches "DARIJGPT"
2. Version is supported
3. Config matches the current model architecture

If any check fails, loading aborts to prevent crashes from incompatible files.

---

## Phase 17: Chat Mode

### Goal
Talk to the model in a terminal.

### Files
- `src/chat.cpp`

### How It Works

```
1. Load saved model from models/darija_gpt.bin
2. Loop:
   a. Read user input
   b. Encode to tokens and add to history
   c. Truncate history to context_length
   d. Generate response tokens
   e. Decode response and print
```

### Commands

| Command | Effect |
|---|---|
| Type anything | Model generates a response |
| `/quit` | Exit the chat |
| `/temp 0.8` | Set temperature (0.0 = greedy, higher = more random) |
| `/len 100` | Set maximum response length |

### Temperature Example

```
User: salam
Bot (temp=0.0):  salam 3likom
Bot (temp=0.8):  salam ya sadiqi
Bot (temp=1.5):  salam wa7ed jouj tlata
```

Lower temperature = safer, more predictable responses.
Higher temperature = more creative, sometimes nonsensical responses.

---

## Quick Glossary

| Term | Meaning |
|---|---|---|
| **Token** | A number representing a piece of text (byte or word) |
| **Embedding** | A vector (list of numbers) representing a token |
| **Logit** | Raw prediction score before softmax |
| **Softmax** | Converts scores to probabilities that sum to 1 |
| **Loss** | Number measuring how wrong the model is (lower = better) |
| **Gradient** | Direction and magnitude to change a weight to reduce loss |
| **Backpropagation** | Algorithm to compute gradients backward through the network |
| **Optimizer** | Algorithm to update weights using gradients (SGD, Adam, etc.) |
| **Batch** | A group of training examples processed together |
| **Epoch** | One pass through the entire dataset |
| **Learning Rate** | Step size for weight updates (0.01 in our code) |
| **Forward Pass** | Computing predictions from input |
| **Backward Pass** | Computing gradients from loss |
| **Temperature** | Controls randomness in text generation (0 = greedy, higher = more random) |
| **Attention** | Mechanism allowing tokens to look at each other |
| **Query (Q)** | What a token is looking for |
| **Key (K)** | What a token contains |
| **Value (V)** | Information a token provides |
| **Causal Mask** | Prevents looking at future tokens during generation |
| **Head** | One independent attention computation within multi-head attention |
| **Head Dimension** | Size of each head's embedding slice (embed_dim / num_heads) |
| **Feed-Forward Network** | Two linear layers with activation that processes each token independently |
| **GELU** | Smooth activation function: 0.5 * x * (1 + erf(x / sqrt(2))) |
| **LayerNorm** | Normalizes values across embedding dimension per token |
| **Residual Connection** | Skip connection that adds input to output: output = input + layer(input) |
| **TransformerBlock** | Complete unit with Attention + FeedForward + LayerNorm + Residuals |
| **Pre-Norm** | Architecture that applies LayerNorm before each sublayer |
| **GPT** | Generative Pre-trained Transformer - the full language model |
| **Generate** | Produce new text by repeatedly predicting the next token |
| **Save/Load** | Store and retrieve trained model weights from disk |
| **Chat Mode** | Interactive terminal program to talk with the model |
| **Overfitting** | When model memorizes training data but fails on new data |

---

*Document updated for Tiny Darija GPT project. Explains Phases 1-17 with concrete examples.*
