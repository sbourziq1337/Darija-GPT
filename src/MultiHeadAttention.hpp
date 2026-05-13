#pragma once

#include <fstream>
#include <vector>

// Multi-head self-attention: splits embedding dimension into multiple heads,
// runs causal self-attention in parallel, concatenates results, and applies output projection
class MultiHeadAttention {
private:
    int embed_dim;
    int num_heads;
    int head_dim;
    float scale;

    // Combined Q/K/V projection matrices: embed_dim x embed_dim
    // Each head gets a slice of embed_dim columns (head_dim wide)
    std::vector<std::vector<float>> Wq;
    std::vector<std::vector<float>> Wk;
    std::vector<std::vector<float>> Wv;

    // Output projection: embed_dim x embed_dim
    std::vector<std::vector<float>> Wo;

    // Gradients for all projection matrices
    std::vector<std::vector<float>> grad_Wq;
    std::vector<std::vector<float>> grad_Wk;
    std::vector<std::vector<float>> grad_Wv;
    std::vector<std::vector<float>> grad_Wo;

    // Cached values from forward for backward pass
    // Q, K, V: shape (batch, seq, embed_dim)
    std::vector<std::vector<std::vector<float>>> cache_Q;
    std::vector<std::vector<std::vector<float>>> cache_K;
    std::vector<std::vector<std::vector<float>>> cache_V;
    // Per-head attention weights: shape (batch, head, seq, seq)
    std::vector<std::vector<std::vector<std::vector<float>>>> cache_weights;
    // Concatenated head outputs before Wo: shape (batch, seq, embed_dim)
    std::vector<std::vector<std::vector<float>>> cache_concat;

public:
    // Initializes Q, K, V, and output projection matrices with small random values
    MultiHeadAttention(int embed_dim, int num_heads);

    // Computes multi-head causal self-attention
    // Splits Q/K/V into heads, attends, concatenates, applies Wo
    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<std::vector<std::vector<float>>>& input
    );

    // Resets gradients for Wq, Wk, Wv, Wo to zero
    void zero_grad();

    // Computes gradients for all projections and returns gradient w.r.t. input
    std::vector<std::vector<std::vector<float>>> backward(
        const std::vector<std::vector<std::vector<float>>>& input,
        const std::vector<std::vector<std::vector<float>>>& grad_output
    );

    // Updates Wq, Wk, Wv, Wo using SGD
    void apply_gradients(float learning_rate);

    // Serialize weights to binary stream
    void save_weights(std::ofstream& f) const;

    // Load weights from binary stream
    void load_weights(std::ifstream& f);
};
