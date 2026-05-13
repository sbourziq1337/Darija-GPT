#include "Dataset.hpp"

#include <algorithm>
#include <stdexcept>

// Splits tokens into train/validation and initializes random generator
Dataset::Dataset(const std::vector<int>& tokens, int batch_size, int block_size, double train_split)
    : batch_size(batch_size),
      block_size(block_size),
      rng(std::random_device{}()) {
    if (batch_size <= 0) {
        throw std::runtime_error("batch_size must be greater than zero");
    }

    if (block_size <= 0) {
        throw std::runtime_error("block_size must be greater than zero");
    }

    if (train_split <= 0.0 || train_split >= 1.0) {
        throw std::runtime_error("train_split must be between 0 and 1");
    }

    if (tokens.size() < static_cast<std::size_t>(block_size + 2)) {
        throw std::runtime_error("Dataset is too small for this block_size");
    }

    std::size_t split_index = static_cast<std::size_t>(tokens.size() * train_split);
    split_index = std::clamp<std::size_t>(split_index, 1, tokens.size() - 1);

    train_tokens.assign(tokens.begin(), tokens.begin() + split_index);
    validation_tokens.assign(tokens.begin() + split_index, tokens.end());

    // Ensure each split is large enough to form at least one batch
    std::size_t min_tokens = static_cast<std::size_t>(block_size) + 2;
    if (train_tokens.size() < min_tokens) {
        throw std::runtime_error("Training split is too small for this block_size");
    }
    if (validation_tokens.size() < min_tokens) {
        throw std::runtime_error("Validation split is too small for this block_size");
    }
}

// Creates random batches where target is input shifted by one position
Batch Dataset::get_batch(Split split) {
    const std::vector<int>& tokens = (split == Split::Train) ? train_tokens : validation_tokens;
    Batch batch;

    int max_start = static_cast<int>(tokens.size()) - block_size - 1;

    if (max_start < 0) {
        throw std::runtime_error("Selected split is too small for this block_size");
    }

    std::uniform_int_distribution<int> dist(0, max_start);
    batch.inputs.reserve(batch_size);
    batch.targets.reserve(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        int start = dist(rng);

        std::vector<int> input_row;
        std::vector<int> target_row;
        input_row.reserve(block_size);
        target_row.reserve(block_size);

        for (int i = 0; i < block_size; ++i) {
            input_row.push_back(tokens[start + i]);
            target_row.push_back(tokens[start + i + 1]);
        }

        batch.inputs.push_back(input_row);
        batch.targets.push_back(target_row);
    }

    return batch;
}
