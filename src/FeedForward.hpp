#pragma once

#include <fstream>
#include <vector>
#include "Linear.hpp"

// Feed-forward network: Linear(up) -> GELU -> Linear(down)
class FeedForward {
private:
    Linear linear_up;    // embed_dim -> 4 * embed_dim
    Linear linear_down;  // 4 * embed_dim -> embed_dim

    // Cached values for backward pass
    std::vector<std::vector<std::vector<float>>> cache_preact;  // before GELU
    std::vector<std::vector<std::vector<float>>> cache_postact; // after GELU

    static float gelu(float x);
    static float gelu_derivative(float x);

public:
    explicit FeedForward(int embed_dim);

    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<std::vector<std::vector<float>>>& input
    );

    void zero_grad();

    std::vector<std::vector<std::vector<float>>> backward(
        const std::vector<std::vector<std::vector<float>>>& input,
        const std::vector<std::vector<std::vector<float>>>& grad_output
    );

    void apply_gradients(float learning_rate);

    // Serialize weights to binary stream
    void save_weights(std::ofstream& f) const;

    // Load weights from binary stream
    void load_weights(std::ifstream& f);
};
