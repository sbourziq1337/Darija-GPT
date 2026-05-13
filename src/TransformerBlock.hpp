#pragma once

#include <fstream>
#include <vector>
#include "LayerNorm.hpp"
#include "MultiHeadAttention.hpp"
#include "FeedForward.hpp"

// Transformer block: LayerNorm -> Attention -> Residual -> LayerNorm -> FeedForward -> Residual
class TransformerBlock {
private:
    LayerNorm ln1;
    MultiHeadAttention attn;
    LayerNorm ln2;
    FeedForward ff;

    // Caches for backward pass
    std::vector<std::vector<std::vector<float>>> cache_input;
    std::vector<std::vector<std::vector<float>>> cache_ln1_out;
    std::vector<std::vector<std::vector<float>>> cache_after_attn;
    std::vector<std::vector<std::vector<float>>> cache_ln2_out;

public:
    TransformerBlock(int embed_dim, int num_heads);

    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<std::vector<std::vector<float>>>& input
    );

    void zero_grad();

    std::vector<std::vector<std::vector<float>>> backward(
        const std::vector<std::vector<std::vector<float>>>& grad_output
    );

    void apply_gradients(float learning_rate);

    // Serialize weights to binary stream
    void save_weights(std::ofstream& f) const;

    // Load weights from binary stream
    void load_weights(std::ifstream& f);
};
