#include "Linear.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

// Creates weight matrix with random values and zero bias
Linear::Linear(int input_dim, int output_dim, float init_std)
    : input_dim(input_dim), output_dim(output_dim) {
    if (input_dim <= 0) {
        throw std::runtime_error("input_dim must be greater than zero");
    }

    if (output_dim <= 0) {
        throw std::runtime_error("output_dim must be greater than zero");
    }

    std::mt19937 rng(std::random_device{}());
    float init_range;
    if (init_std > 0.0f) {
        init_range = init_std;
    } else {
        init_range = std::sqrt(6.0f / (input_dim + output_dim));
    }
    std::uniform_real_distribution<float> dist(-init_range, init_range);

    weights.resize(input_dim);
    for (int in = 0; in < input_dim; ++in) {
        weights[in].resize(output_dim);

        for (int out = 0; out < output_dim; ++out) {
            weights[in][out] = dist(rng);
        }
    }

    bias.resize(output_dim, 0.0f);

    grad_weights.resize(input_dim);
    for (int in = 0; in < input_dim; ++in) {
        grad_weights[in].resize(output_dim, 0.0f);
    }

    grad_bias.resize(output_dim, 0.0f);
}

// For each input vector: output[out] = bias[out] + sum(input[in] * weights[in][out])
std::vector<std::vector<std::vector<float>>> Linear::forward(
    const std::vector<std::vector<std::vector<float>>>& input_batch
) const {
    std::vector<std::vector<std::vector<float>>> output;
    output.reserve(input_batch.size());

    for (const auto& input_sequence : input_batch) {
        std::vector<std::vector<float>> output_sequence;
        output_sequence.reserve(input_sequence.size());

        for (const auto& input_vector : input_sequence) {
            if (static_cast<int>(input_vector.size()) != input_dim) {
                throw std::runtime_error("Input vector size does not match Linear input_dim");
            }

            std::vector<float> output_vector(output_dim);

            for (int out = 0; out < output_dim; ++out) {
                float score = bias[out];

                for (int in = 0; in < input_dim; ++in) {
                    score += input_vector[in] * weights[in][out];
                }

                output_vector[out] = score;
            }

            output_sequence.push_back(std::move(output_vector));
        }

        output.push_back(std::move(output_sequence));
    }

    return output;
}

// Sets all weight and bias gradients to zero
void Linear::zero_grad() {
    for (int in = 0; in < input_dim; ++in) {
        for (int out = 0; out < output_dim; ++out) {
            grad_weights[in][out] = 0.0f;
        }
    }

    for (int out = 0; out < output_dim; ++out) {
        grad_bias[out] = 0.0f;
    }
}

// Computes grad_weights, grad_bias, and returns grad_input for previous layer
std::vector<std::vector<std::vector<float>>> Linear::backward(
    const std::vector<std::vector<std::vector<float>>>& input,
    const std::vector<std::vector<std::vector<float>>>& grad_output
) {
    if (input.empty() || grad_output.empty()) {
        throw std::runtime_error("Linear::backward requires non-empty input and grad_output");
    }

    if (input.size() != grad_output.size()) {
        throw std::runtime_error("Linear::backward: input and grad_output batch sizes must match");
    }

    std::vector<std::vector<std::vector<float>>> grad_input(
        input.size(),
        std::vector<std::vector<float>>(
            input[0].size(),
            std::vector<float>(input_dim, 0.0f)
        )
    );

    for (std::size_t b = 0; b < input.size(); ++b) {
        if (input[b].size() != grad_output[b].size()) {
            throw std::runtime_error("Linear::backward: sequence lengths must match");
        }

        for (std::size_t t = 0; t < input[b].size(); ++t) {
            if (static_cast<int>(input[b][t].size()) != input_dim) {
                throw std::runtime_error("Linear::backward: input vector size does not match input_dim");
            }

            if (static_cast<int>(grad_output[b][t].size()) != output_dim) {
                throw std::runtime_error("Linear::backward: grad_output vector size does not match output_dim");
            }

            for (int out = 0; out < output_dim; ++out) {
                float go = grad_output[b][t][out];
                grad_bias[out] += go;

                for (int in = 0; in < input_dim; ++in) {
                    grad_weights[in][out] += input[b][t][in] * go;
                    grad_input[b][t][in] += go * weights[in][out];
                }
            }
        }
    }

    return grad_input;
}

// Updates weights and bias: param = param - learning_rate * gradient
void Linear::apply_gradients(float learning_rate) {
    // Gradient clipping (per-layer L2 norm)
    const float max_norm = 1.0f;
    float grad_norm_sq = 0.0f;
    for (int in = 0; in < input_dim; ++in) {
        for (int out = 0; out < output_dim; ++out) {
            grad_norm_sq += grad_weights[in][out] * grad_weights[in][out];
        }
    }
    for (int out = 0; out < output_dim; ++out) {
        grad_norm_sq += grad_bias[out] * grad_bias[out];
    }
    float grad_norm = std::sqrt(grad_norm_sq);
    float scale = (grad_norm > max_norm) ? (max_norm / grad_norm) : 1.0f;

    for (int in = 0; in < input_dim; ++in) {
        for (int out = 0; out < output_dim; ++out) {
            weights[in][out] -= learning_rate * grad_weights[in][out] * scale;
        }
    }

    for (int out = 0; out < output_dim; ++out) {
        bias[out] -= learning_rate * grad_bias[out] * scale;
    }
}

void Linear::save_weights(std::ostream& f) const {
    for (int in = 0; in < input_dim; ++in) {
        f.write(reinterpret_cast<const char*>(weights[in].data()),
                output_dim * sizeof(float));
    }
    f.write(reinterpret_cast<const char*>(bias.data()),
            output_dim * sizeof(float));
}

void Linear::load_weights(std::ifstream& f) {
    for (int in = 0; in < input_dim; ++in) {
        f.read(reinterpret_cast<char*>(weights[in].data()),
               output_dim * sizeof(float));
    }
    f.read(reinterpret_cast<char*>(bias.data()),
           output_dim * sizeof(float));
}
