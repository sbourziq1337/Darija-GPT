#pragma once

#include <fstream>
#include <vector>

// Layer normalization: normalizes across the embedding dimension per token
// y = (x - mean) / sqrt(var + eps) * gamma + beta
class LayerNorm {
private:
    int embed_dim;
    float eps;

    std::vector<float> gamma;
    std::vector<float> beta;
    std::vector<float> grad_gamma;
    std::vector<float> grad_beta;

    // Caches for backward pass
    std::vector<std::vector<std::vector<float>>> cache_x;
    std::vector<std::vector<std::vector<float>>> cache_x_hat;
    std::vector<std::vector<float>> cache_mean;
    std::vector<std::vector<float>> cache_std;

public:
    LayerNorm(int embed_dim, float eps = 1e-5f);

    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<std::vector<std::vector<float>>>& input
    );

    void zero_grad();

    // Given dL/dy, computes dL/dx and accumulates dL/dgamma, dL/dbeta
    std::vector<std::vector<std::vector<float>>> backward(
        const std::vector<std::vector<std::vector<float>>>& grad_output
    );

    void apply_gradients(float learning_rate);

    // Serialize weights to binary stream
    void save_weights(std::ostream& f) const;

    // Load weights from binary stream
    void load_weights(std::ifstream& f);
};
