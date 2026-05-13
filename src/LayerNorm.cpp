#include "LayerNorm.hpp"

#include <cmath>
#include <stdexcept>

LayerNorm::LayerNorm(int embed_dim, float eps)
    : embed_dim(embed_dim), eps(eps) {
    if (embed_dim <= 0) {
        throw std::runtime_error("embed_dim must be greater than zero");
    }

    gamma.resize(embed_dim, 1.0f);
    beta.resize(embed_dim, 0.0f);
    grad_gamma.resize(embed_dim, 0.0f);
    grad_beta.resize(embed_dim, 0.0f);
}

std::vector<std::vector<std::vector<float>>> LayerNorm::forward(
    const std::vector<std::vector<std::vector<float>>>& input
) {
    std::size_t batch_size = input.size();
    if (batch_size == 0) {
        return {};
    }

    std::size_t seq_len = input[0].size();

    cache_x = input;
    cache_x_hat.resize(batch_size);
    cache_mean.resize(batch_size);
    cache_std.resize(batch_size);

    std::vector<std::vector<std::vector<float>>> output(batch_size);

    for (std::size_t b = 0; b < batch_size; ++b) {
        if (input[b].size() != seq_len) {
            throw std::runtime_error("LayerNorm::forward: inconsistent sequence lengths");
        }

        cache_x_hat[b].resize(seq_len);
        cache_mean[b].resize(seq_len);
        cache_std[b].resize(seq_len);
        output[b].resize(seq_len);

        for (std::size_t t = 0; t < seq_len; ++t) {
            if (static_cast<int>(input[b][t].size()) != embed_dim) {
                throw std::runtime_error("LayerNorm::forward: input dimension mismatch");
            }

            // Compute mean
            float mean = 0.0f;
            for (int d = 0; d < embed_dim; ++d) {
                mean += input[b][t][d];
            }
            mean /= embed_dim;
            cache_mean[b][t] = mean;

            // Compute variance
            float var = 0.0f;
            for (int d = 0; d < embed_dim; ++d) {
                float diff = input[b][t][d] - mean;
                var += diff * diff;
            }
            var /= embed_dim;

            // Compute std
            float std = std::sqrt(var + eps);
            cache_std[b][t] = std;

            // Normalize and apply gamma/beta
            output[b][t].resize(embed_dim);
            cache_x_hat[b][t].resize(embed_dim);
            for (int d = 0; d < embed_dim; ++d) {
                float x_hat = (input[b][t][d] - mean) / std;
                cache_x_hat[b][t][d] = x_hat;
                output[b][t][d] = x_hat * gamma[d] + beta[d];
            }
        }
    }

    return output;
}

void LayerNorm::zero_grad() {
    for (int d = 0; d < embed_dim; ++d) {
        grad_gamma[d] = 0.0f;
        grad_beta[d] = 0.0f;
    }
}

std::vector<std::vector<std::vector<float>>> LayerNorm::backward(
    const std::vector<std::vector<std::vector<float>>>& grad_output
) {
    std::size_t batch_size = grad_output.size();
    if (batch_size == 0) {
        return {};
    }

    std::size_t seq_len = grad_output[0].size();

    std::vector<std::vector<std::vector<float>>> grad_input(batch_size);

    for (std::size_t b = 0; b < batch_size; ++b) {
        if (grad_output[b].size() != seq_len) {
            throw std::runtime_error("LayerNorm::backward: inconsistent sequence lengths");
        }

        grad_input[b].resize(seq_len);

        for (std::size_t t = 0; t < seq_len; ++t) {
            if (static_cast<int>(grad_output[b][t].size()) != embed_dim) {
                throw std::runtime_error("LayerNorm::backward: grad_output dimension mismatch");
            }

            // Compute grad_gamma and grad_beta
            for (int d = 0; d < embed_dim; ++d) {
                grad_gamma[d] += grad_output[b][t][d] * cache_x_hat[b][t][d];
                grad_beta[d] += grad_output[b][t][d];
            }

            // Compute grad_x_hat
            std::vector<float> grad_x_hat(embed_dim);
            for (int d = 0; d < embed_dim; ++d) {
                grad_x_hat[d] = grad_output[b][t][d] * gamma[d];
            }

            // LayerNorm backward formula:
            // dL/dx = (1/std) * (dL/dx_hat - mean(dL/dx_hat) - x_hat * mean(dL/dx_hat * x_hat))
            float mean_grad_x_hat = 0.0f;
            float mean_grad_x_hat_x_hat = 0.0f;
            for (int d = 0; d < embed_dim; ++d) {
                mean_grad_x_hat += grad_x_hat[d];
                mean_grad_x_hat_x_hat += grad_x_hat[d] * cache_x_hat[b][t][d];
            }
            mean_grad_x_hat /= embed_dim;
            mean_grad_x_hat_x_hat /= embed_dim;

            float std = cache_std[b][t];
            grad_input[b][t].resize(embed_dim);
            for (int d = 0; d < embed_dim; ++d) {
                grad_input[b][t][d] = (grad_x_hat[d] - mean_grad_x_hat - cache_x_hat[b][t][d] * mean_grad_x_hat_x_hat) / std;
            }
        }
    }

    return grad_input;
}

void LayerNorm::apply_gradients(float learning_rate) {
    // Gradient clipping (per-layer L2 norm)
    const float max_norm = 1.0f;
    float grad_norm_sq = 0.0f;
    for (int d = 0; d < embed_dim; ++d) {
        grad_norm_sq += grad_gamma[d] * grad_gamma[d];
        grad_norm_sq += grad_beta[d] * grad_beta[d];
    }
    float grad_norm = std::sqrt(grad_norm_sq);
    float scale = (grad_norm > max_norm) ? (max_norm / grad_norm) : 1.0f;

    for (int d = 0; d < embed_dim; ++d) {
        gamma[d] -= learning_rate * grad_gamma[d] * scale;
        beta[d] -= learning_rate * grad_beta[d] * scale;
    }
}

void LayerNorm::save_weights(std::ostream& f) const {
    f.write(reinterpret_cast<const char*>(gamma.data()), embed_dim * sizeof(float));
    f.write(reinterpret_cast<const char*>(beta.data()), embed_dim * sizeof(float));
}

void LayerNorm::load_weights(std::ifstream& f) {
    f.read(reinterpret_cast<char*>(gamma.data()), embed_dim * sizeof(float));
    f.read(reinterpret_cast<char*>(beta.data()), embed_dim * sizeof(float));
}
