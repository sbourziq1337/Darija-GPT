#pragma once

#include <fstream>
#include <vector>

// Converts token IDs to vectors using token embeddings + position embeddings
class Embedding {
private:
    int vocab_size;
    int embedding_dim;
    int context_length;
    std::vector<std::vector<float>> token_embeddings;
    std::vector<std::vector<float>> position_embeddings;
    std::vector<std::vector<float>> grad_token_embeddings;
    std::vector<std::vector<float>> grad_position_embeddings;

public:
    // Initializes token and position embedding tables with small random values
    Embedding(int vocab_size, int embedding_dim, int context_length);

    // Looks up token embeddings and adds position embeddings
    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<std::vector<int>>& input_batch
    ) const;

    // Resets all gradients to zero before each backward pass
    void zero_grad();

    // Accumulates gradients into token and position embedding tables
    void backward(
        const std::vector<std::vector<int>>& input_batch,
        const std::vector<std::vector<std::vector<float>>>& grad_output
    );

    // Updates embeddings using SGD: param = param - lr * gradient
    void apply_gradients(float learning_rate);

    // Serialize weights to binary stream
    void save_weights(std::ostream& f) const;

    // Load weights from binary stream
    void load_weights(std::ifstream& f);
};
