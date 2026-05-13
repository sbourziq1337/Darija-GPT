#pragma once

#include <string>
#include <vector>
#include "ByteTokenizer.hpp"
#include "Checkpoint.hpp"
#include "Dataset.hpp"
#include "Embedding.hpp"
#include "LayerNorm.hpp"
#include "Linear.hpp"
#include "TransformerBlock.hpp"

// GPT-style language model: Embedding -> TransformerBlocks -> LayerNorm -> Linear
class GPT {
private:
    // Model config
    int vocab_size_;
    int context_length_;
    int embed_dim_;
    int num_heads_;
    int num_layers_;

    // Layers
    Embedding embedding_;
    std::vector<TransformerBlock> blocks_;
    LayerNorm final_ln_;
    Linear lm_head_;

    // Cached values for backward/generation
    std::vector<std::vector<int>> cache_input_ids_;
    std::vector<std::vector<std::vector<float>>> cache_embedded_;
    std::vector<std::vector<std::vector<float>>> cache_final_ln_;

public:
    GPT(int vocab_size, int context_length, int embed_dim, int num_heads, int num_layers);

    // Forward pass: token IDs -> logits [B][T][V]
    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<std::vector<int>>& input_batch
    );

    // Backward pass: computes gradients for all layers
    void backward(
        const std::vector<std::vector<int>>& input_batch,
        const std::vector<std::vector<std::vector<float>>>& grad_logits
    );

    // Zero all gradients
    void zero_grad();

    // Apply SGD updates
    void apply_gradients(float learning_rate);

    // Generate text autoregressively
    std::vector<int> generate(
        const std::vector<int>& prompt,
        int max_tokens,
        float temperature
    );

    // Save weights to binary file (version 1 compatible)
    bool save(const std::string& path) const;

    // Save weights with checkpoint metadata (version 2)
    bool save(const std::string& path, const CheckpointMetadata& meta) const;

    // Load weights from binary file
    bool load(const std::string& path);

    // Load weights and extract checkpoint metadata (version 2)
    bool load(const std::string& path, CheckpointMetadata& out_meta);

    // Getters
    int vocab_size() const { return vocab_size_; }
    int context_length() const { return context_length_; }
    int embed_dim() const { return embed_dim_; }
    int num_heads() const { return num_heads_; }
    int num_layers() const { return num_layers_; }
};
