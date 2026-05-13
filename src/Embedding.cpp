#include "Embedding.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

// Creates token and position embedding tables with random values (-0.01 to 0.01)
Embedding::Embedding(int vocab_size, int embedding_dim, int context_length)
    : vocab_size(vocab_size), embedding_dim(embedding_dim), context_length(context_length) {
    if (vocab_size <= 0) {
        throw std::runtime_error("vocab_size must be greater than zero");
    }

    if (embedding_dim <= 0) {
        throw std::runtime_error("embedding_dim must be greater than zero");
    }

    if (context_length <= 0) {
        throw std::runtime_error("context_length must be greater than zero");
    }

    std::mt19937 rng(std::random_device{}());
    float init_range = std::sqrt(1.0f / embedding_dim);
    std::uniform_real_distribution<float> dist(-init_range, init_range);

    token_embeddings.resize(vocab_size);

    for (int i = 0; i < vocab_size; ++i) {
        token_embeddings[i].resize(embedding_dim);

        for (int j = 0; j < embedding_dim; ++j) {
            token_embeddings[i][j] = dist(rng);
        }
    }

    position_embeddings.resize(context_length);
    grad_token_embeddings.resize(vocab_size);
    grad_position_embeddings.resize(context_length);

    for (int pos = 0; pos < context_length; ++pos) {
        position_embeddings[pos].resize(embedding_dim);
        grad_position_embeddings[pos].resize(embedding_dim, 0.0f);

        for (int j = 0; j < embedding_dim; ++j) {
            position_embeddings[pos][j] = dist(rng);
        }
    }

    for (int i = 0; i < vocab_size; ++i) {
        grad_token_embeddings[i].resize(embedding_dim, 0.0f);
    }
}

// For each token: final_embedding = token_embedding[token_id] + position_embedding[position]
std::vector<std::vector<std::vector<float>>> Embedding::forward(
    const std::vector<std::vector<int>>& input_batch
) const {
    std::vector<std::vector<std::vector<float>>> output;
    output.reserve(input_batch.size());

    for (const auto& input_row : input_batch) {
        if (static_cast<int>(input_row.size()) > context_length) {
            throw std::runtime_error("Input row is longer than context_length");
        }

        std::vector<std::vector<float>> embedded_row;
        embedded_row.reserve(input_row.size());

        for (std::size_t position = 0; position < input_row.size(); ++position) {
            int token_id = input_row[position];

            if (token_id < 0 || token_id >= vocab_size) {
                throw std::runtime_error("Token ID out of bounds in forward pass");
            }

            std::vector<float> final_embedding;
            final_embedding.reserve(embedding_dim);

            for (int dim = 0; dim < embedding_dim; ++dim) {
                final_embedding.push_back(
                    token_embeddings[token_id][dim] + position_embeddings[position][dim]
                );
            }

            embedded_row.push_back(std::move(final_embedding));
        }

        output.push_back(std::move(embedded_row));
    }

    return output;
}

// Sets all gradient values to zero
void Embedding::zero_grad() {
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < embedding_dim; ++j) {
            grad_token_embeddings[i][j] = 0.0f;
        }
    }

    for (int pos = 0; pos < context_length; ++pos) {
        for (int j = 0; j < embedding_dim; ++j) {
            grad_position_embeddings[pos][j] = 0.0f;
        }
    }
}

// Accumulates grad_output into token and position gradient tables using +=
void Embedding::backward(
    const std::vector<std::vector<int>>& input_batch,
    const std::vector<std::vector<std::vector<float>>>& grad_output
) {
    if (input_batch.size() != grad_output.size()) {
        throw std::runtime_error("Embedding::backward: batch sizes must match");
    }

    for (std::size_t b = 0; b < input_batch.size(); ++b) {
        if (input_batch[b].size() != grad_output[b].size()) {
            throw std::runtime_error("Embedding::backward: sequence lengths must match");
        }

        for (std::size_t t = 0; t < input_batch[b].size(); ++t) {
            int token_id = input_batch[b][t];

            if (token_id < 0 || token_id >= vocab_size) {
                throw std::runtime_error("Token ID out of bounds in Embedding::backward");
            }

            if (static_cast<int>(grad_output[b][t].size()) != embedding_dim) {
                throw std::runtime_error("grad_output dimension does not match embedding_dim in Embedding::backward");
            }

            for (int j = 0; j < embedding_dim; ++j) {
                grad_token_embeddings[token_id][j] += grad_output[b][t][j];
                grad_position_embeddings[t][j] += grad_output[b][t][j];
            }
        }
    }
}

// Updates token and position embeddings using SGD
void Embedding::apply_gradients(float learning_rate) {
    // Gradient clipping (per-layer L2 norm)
    const float max_norm = 1.0f;
    float grad_norm_sq = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < embedding_dim; ++j) {
            grad_norm_sq += grad_token_embeddings[i][j] * grad_token_embeddings[i][j];
        }
    }
    for (int pos = 0; pos < context_length; ++pos) {
        for (int j = 0; j < embedding_dim; ++j) {
            grad_norm_sq += grad_position_embeddings[pos][j] * grad_position_embeddings[pos][j];
        }
    }
    float grad_norm = std::sqrt(grad_norm_sq);
    float scale = (grad_norm > max_norm) ? (max_norm / grad_norm) : 1.0f;

    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < embedding_dim; ++j) {
            token_embeddings[i][j] -= learning_rate * grad_token_embeddings[i][j] * scale;
        }
    }

    for (int pos = 0; pos < context_length; ++pos) {
        for (int j = 0; j < embedding_dim; ++j) {
            position_embeddings[pos][j] -= learning_rate * grad_position_embeddings[pos][j] * scale;
        }
    }
}

// Serialize weights
void Embedding::save_weights(std::ostream& f) const {
    for (int i = 0; i < vocab_size; ++i) {
        f.write(reinterpret_cast<const char*>(token_embeddings[i].data()),
                embedding_dim * sizeof(float));
    }
    for (int pos = 0; pos < context_length; ++pos) {
        f.write(reinterpret_cast<const char*>(position_embeddings[pos].data()),
                embedding_dim * sizeof(float));
    }
}

// Load weights
void Embedding::load_weights(std::ifstream& f) {
    for (int i = 0; i < vocab_size; ++i) {
        f.read(reinterpret_cast<char*>(token_embeddings[i].data()),
               embedding_dim * sizeof(float));
    }
    for (int pos = 0; pos < context_length; ++pos) {
        f.read(reinterpret_cast<char*>(position_embeddings[pos].data()),
               embedding_dim * sizeof(float));
    }
}
