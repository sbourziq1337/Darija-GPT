#pragma once

#include <fstream>
#include <vector>

// Linear layer: computes output = input * weights + bias
class Linear {
private:
    int input_dim;
    int output_dim;
    std::vector<std::vector<float>> weights;
    std::vector<float> bias;
    std::vector<std::vector<float>> grad_weights;
    std::vector<float> grad_bias;

public:
    // Initializes weights with small random values and bias to zero
    Linear(int input_dim, int output_dim);

    // Computes matrix multiplication: output = input * weights + bias
    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<std::vector<std::vector<float>>>& input_batch
    ) const;

    // Resets weight and bias gradients to zero
    void zero_grad();

    // Computes gradients for weights, bias, and returns gradient w.r.t. input
    std::vector<std::vector<std::vector<float>>> backward(
        const std::vector<std::vector<std::vector<float>>>& input,
        const std::vector<std::vector<std::vector<float>>>& grad_output
    );

    // Updates weights and bias using SGD
    void apply_gradients(float learning_rate);

    // Serialize weights to binary stream
    void save_weights(std::ofstream& f) const;

    // Load weights from binary stream
    void load_weights(std::ifstream& f);
};
