#pragma once

#include <random>
#include <vector>

// Dataset split type: Train (90%) or Validation (10%)
enum class Split {
    Train,
    Validation
};

// Batch containing input sequences and target sequences (shifted by +1)
struct Batch {
    std::vector<std::vector<int>> inputs;
    std::vector<std::vector<int>> targets;
};

// Manages train/validation split and creates random batches for training
class Dataset {
private:
    int batch_size;
    int block_size;
    std::mt19937 rng;
    std::vector<int> train_tokens;
    std::vector<int> validation_tokens;

public:
    // Splits tokens into train (90%) and validation (10%)
    Dataset(const std::vector<int>& tokens, int batch_size, int block_size, double train_split = 0.9);

    // Returns a random batch from specified split
    Batch get_batch(Split split = Split::Train);
};
