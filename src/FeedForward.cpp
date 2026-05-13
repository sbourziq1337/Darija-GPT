#include "FeedForward.hpp"

#include <cmath>
#include <stdexcept>

namespace {
    const float SQRT_2 = std::sqrt(2.0f);
    const float SQRT_2PI = std::sqrt(2.0f * 3.14159265358979323846f);
}

float FeedForward::gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x / SQRT_2));
}

float FeedForward::gelu_derivative(float x) {
    float phi = 0.5f * (1.0f + std::erf(x / SQRT_2));
    float pdf = std::exp(-0.5f * x * x) / SQRT_2PI;
    return phi + x * pdf;
}

FeedForward::FeedForward(int embed_dim)
    : linear_up(embed_dim, 4 * embed_dim),
      linear_down(4 * embed_dim, embed_dim) {
    if (embed_dim <= 0) {
        throw std::runtime_error("embed_dim must be greater than zero");
    }
}

std::vector<std::vector<std::vector<float>>> FeedForward::forward(
    const std::vector<std::vector<std::vector<float>>>& input
) {
    // Up-projection: embed_dim -> 4 * embed_dim
    auto up = linear_up.forward(input);

    cache_preact = up;
    cache_postact.resize(up.size());

    for (std::size_t b = 0; b < up.size(); ++b) {
        cache_postact[b].resize(up[b].size());
        for (std::size_t t = 0; t < up[b].size(); ++t) {
            cache_postact[b][t].resize(up[b][t].size());
            for (std::size_t d = 0; d < up[b][t].size(); ++d) {
                cache_postact[b][t][d] = gelu(up[b][t][d]);
            }
        }
    }

    // Down-projection: 4 * embed_dim -> embed_dim
    return linear_down.forward(cache_postact);
}

void FeedForward::zero_grad() {
    linear_up.zero_grad();
    linear_down.zero_grad();
}

std::vector<std::vector<std::vector<float>>> FeedForward::backward(
    const std::vector<std::vector<std::vector<float>>>& input,
    const std::vector<std::vector<std::vector<float>>>& grad_output
) {
    if (input.empty() || grad_output.empty()) {
        throw std::runtime_error("FeedForward::backward requires non-empty input and grad_output");
    }
    if (input.size() != grad_output.size()) {
        throw std::runtime_error("FeedForward::backward: batch sizes must match");
    }

    // Backprop through linear_down (input = cache_postact)
    auto grad_postact = linear_down.backward(cache_postact, grad_output);

    if (grad_postact.size() != cache_preact.size()) {
        throw std::runtime_error("FeedForward::backward: shape mismatch after linear_down");
    }

    // Backprop through GELU
    std::vector<std::vector<std::vector<float>>> grad_preact(grad_postact.size());
    for (std::size_t b = 0; b < grad_postact.size(); ++b) {
        if (grad_postact[b].size() != cache_preact[b].size()) {
            throw std::runtime_error("FeedForward::backward: sequence length mismatch after linear_down");
        }
        grad_preact[b].resize(grad_postact[b].size());
        for (std::size_t t = 0; t < grad_postact[b].size(); ++t) {
            if (grad_postact[b][t].size() != cache_preact[b][t].size()) {
                throw std::runtime_error("FeedForward::backward: dimension mismatch after linear_down");
            }
            grad_preact[b][t].resize(grad_postact[b][t].size());
            for (std::size_t d = 0; d < grad_postact[b][t].size(); ++d) {
                grad_preact[b][t][d] = grad_postact[b][t][d] * gelu_derivative(cache_preact[b][t][d]);
            }
        }
    }

    // Backprop through linear_up
    return linear_up.backward(input, grad_preact);
}

void FeedForward::apply_gradients(float learning_rate) {
    linear_up.apply_gradients(learning_rate);
    linear_down.apply_gradients(learning_rate);
}

void FeedForward::save_weights(std::ostream& f) const {
    linear_up.save_weights(f);
    linear_down.save_weights(f);
}

void FeedForward::load_weights(std::ifstream& f) {
    linear_up.load_weights(f);
    linear_down.load_weights(f);
}
