#include "MultiHeadAttention.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <tuple>

// Runs backward pass for a single attention head
// Returns {grad_Q, grad_K, grad_V} for this head
static std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>, std::vector<std::vector<float>>>
head_backward(
    std::size_t seq_len,
    int head_dim,
    float scale,
    const std::vector<std::vector<float>>& Q,
    const std::vector<std::vector<float>>& K,
    const std::vector<std::vector<float>>& V,
    const std::vector<std::vector<float>>& weights,
    const std::vector<std::vector<float>>& grad_output
) {
    // Step 1: dL/dV = weights^T @ grad_output
    std::vector<std::vector<float>> grad_V(seq_len, std::vector<float>(head_dim, 0.0f));
    for (std::size_t j = 0; j < seq_len; ++j) {
        for (std::size_t i = 0; i < seq_len; ++i) {
            float w = weights[i][j];
            for (int d = 0; d < head_dim; ++d) {
                grad_V[j][d] += w * grad_output[i][d];
            }
        }
    }

    // Step 2: dL/dweights = grad_output @ V^T
    std::vector<std::vector<float>> grad_weights(seq_len, std::vector<float>(seq_len, 0.0f));
    for (std::size_t i = 0; i < seq_len; ++i) {
        for (std::size_t j = 0; j < seq_len; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += grad_output[i][d] * V[j][d];
            }
            grad_weights[i][j] = dot;
        }
    }

    // Step 3: Softmax backward
    // dL/dscores[i][j] = weights[i][j] * (grad_weights[i][j] - sum_k grad_weights[i][k] * weights[i][k])
    std::vector<std::vector<float>> grad_scores(seq_len, std::vector<float>(seq_len, 0.0f));
    for (std::size_t i = 0; i < seq_len; ++i) {
        float weighted_sum = 0.0f;
        for (std::size_t k = 0; k < seq_len; ++k) {
            weighted_sum += grad_weights[i][k] * weights[i][k];
        }
        for (std::size_t j = 0; j < seq_len; ++j) {
            grad_scores[i][j] = weights[i][j] * (grad_weights[i][j] - weighted_sum);
        }
    }

    // Step 4: Scale by 1/sqrt(head_dim)
    for (std::size_t i = 0; i < seq_len; ++i) {
        for (std::size_t j = 0; j < seq_len; ++j) {
            grad_scores[i][j] *= scale;
        }
    }

    // Step 5: dL/dQ = grad_scores @ K, dL/dK = grad_scores^T @ Q
    std::vector<std::vector<float>> grad_Q(seq_len, std::vector<float>(head_dim, 0.0f));
    std::vector<std::vector<float>> grad_K(seq_len, std::vector<float>(head_dim, 0.0f));
    for (std::size_t i = 0; i < seq_len; ++i) {
        for (std::size_t j = 0; j < seq_len; ++j) {
            float gs = grad_scores[i][j];
            for (int d = 0; d < head_dim; ++d) {
                grad_Q[i][d] += gs * K[j][d];
                grad_K[j][d] += gs * Q[i][d];
            }
        }
    }

    return {grad_Q, grad_K, grad_V};
}

// Creates projection matrices with random values (-0.01 to 0.01)
// Validates that embed_dim is divisible by num_heads
MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads)
    : embed_dim(embed_dim),
      num_heads(num_heads),
      head_dim(embed_dim / num_heads),
      scale(1.0f / std::sqrt(static_cast<float>(embed_dim / num_heads))) {

    if (embed_dim <= 0) {
        throw std::runtime_error("embed_dim must be greater than zero");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("num_heads must be greater than zero");
    }
    if (embed_dim % num_heads != 0) {
        throw std::runtime_error("embed_dim must be divisible by num_heads");
    }

    std::mt19937 rng(std::random_device{}());
    float init_range = std::sqrt(6.0f / (embed_dim + embed_dim));
    std::uniform_real_distribution<float> dist(-init_range, init_range);

    // Initialize Wq, Wk, Wv, Wo and their gradients
    Wq.resize(embed_dim);
    Wk.resize(embed_dim);
    Wv.resize(embed_dim);
    Wo.resize(embed_dim);
    grad_Wq.resize(embed_dim);
    grad_Wk.resize(embed_dim);
    grad_Wv.resize(embed_dim);
    grad_Wo.resize(embed_dim);

    for (int i = 0; i < embed_dim; ++i) {
        Wq[i].resize(embed_dim);
        Wk[i].resize(embed_dim);
        Wv[i].resize(embed_dim);
        Wo[i].resize(embed_dim);
        grad_Wq[i].resize(embed_dim, 0.0f);
        grad_Wk[i].resize(embed_dim, 0.0f);
        grad_Wv[i].resize(embed_dim, 0.0f);
        grad_Wo[i].resize(embed_dim, 0.0f);

        for (int j = 0; j < embed_dim; ++j) {
            Wq[i][j] = dist(rng);
            Wk[i][j] = dist(rng);
            Wv[i][j] = dist(rng);
            Wo[i][j] = dist(rng);
        }
    }
}

// Computes multi-head causal self-attention
std::vector<std::vector<std::vector<float>>> MultiHeadAttention::forward(
    const std::vector<std::vector<std::vector<float>>>& input
) {
    std::size_t batch_size = input.size();
    if (batch_size == 0) {
        return {};
    }

    std::size_t seq_len = input[0].size();

    // Resize caches
    cache_Q.resize(batch_size);
    cache_K.resize(batch_size);
    cache_V.resize(batch_size);
    cache_weights.resize(batch_size);
    cache_concat.resize(batch_size);

    std::vector<std::vector<std::vector<float>>> output;
    output.reserve(batch_size);

    // Process each sequence in the batch
    for (std::size_t b = 0; b < batch_size; ++b) {
        if (input[b].size() != seq_len) {
            throw std::runtime_error("MultiHeadAttention::forward: inconsistent sequence lengths");
        }

        // Initialize Q, K, V caches for this batch
        cache_Q[b].resize(seq_len);
        cache_K[b].resize(seq_len);
        cache_V[b].resize(seq_len);
        cache_concat[b].resize(seq_len);
        for (std::size_t i = 0; i < seq_len; ++i) {
            cache_Q[b][i].assign(embed_dim, 0.0f);
            cache_K[b][i].assign(embed_dim, 0.0f);
            cache_V[b][i].assign(embed_dim, 0.0f);
            cache_concat[b][i].assign(embed_dim, 0.0f);
        }

        // Compute Q = X @ Wq, K = X @ Wk, V = X @ Wv
        for (std::size_t t = 0; t < seq_len; ++t) {
            if (static_cast<int>(input[b][t].size()) != embed_dim) {
                throw std::runtime_error("MultiHeadAttention::forward: input dimension mismatch");
            }

            for (int j = 0; j < embed_dim; ++j) {
                float q_val = 0.0f;
                float k_val = 0.0f;
                float v_val = 0.0f;

                for (int i = 0; i < embed_dim; ++i) {
                    q_val += input[b][t][i] * Wq[i][j];
                    k_val += input[b][t][i] * Wk[i][j];
                    v_val += input[b][t][i] * Wv[i][j];
                }

                cache_Q[b][t][j] = q_val;
                cache_K[b][t][j] = k_val;
                cache_V[b][t][j] = v_val;
            }
        }

        // Initialize per-head weights cache
        cache_weights[b].resize(num_heads);
        for (int h = 0; h < num_heads; ++h) {
            cache_weights[b][h].resize(seq_len);
            for (std::size_t i = 0; i < seq_len; ++i) {
                cache_weights[b][h][i].resize(seq_len, 0.0f);
            }
        }

        // Process each attention head (parallelized with OpenMP)
        #pragma omp parallel for
        for (int h = 0; h < num_heads; ++h) {
            int offset = h * head_dim;

            // Compute attention scores for this head
            std::vector<std::vector<float>> scores(seq_len, std::vector<float>(seq_len, 0.0f));

            for (std::size_t i = 0; i < seq_len; ++i) {
                for (std::size_t j = 0; j < seq_len; ++j) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += cache_Q[b][i][offset + d] * cache_K[b][j][offset + d];
                    }
                    scores[i][j] = dot * scale;
                }
            }

            // Apply causal mask: set future positions (j > i) to large negative value
            for (std::size_t i = 0; i < seq_len; ++i) {
                for (std::size_t j = i + 1; j < seq_len; ++j) {
                    scores[i][j] = -1e9f;
                }
            }

            // Softmax over each row
            for (std::size_t i = 0; i < seq_len; ++i) {
                float max_score = scores[i][0];
                for (std::size_t j = 1; j < seq_len; ++j) {
                    if (scores[i][j] > max_score) {
                        max_score = scores[i][j];
                    }
                }

                float sum_exp = 0.0f;
                for (std::size_t j = 0; j < seq_len; ++j) {
                    cache_weights[b][h][i][j] = std::exp(scores[i][j] - max_score);
                    sum_exp += cache_weights[b][h][i][j];
                }

                for (std::size_t j = 0; j < seq_len; ++j) {
                    cache_weights[b][h][i][j] /= sum_exp;
                }
            }

            // Compute weighted sum of values for this head and store in concat
            for (std::size_t i = 0; i < seq_len; ++i) {
                for (std::size_t j = 0; j <= i; ++j) {
                    float w = cache_weights[b][h][i][j];
                    for (int d = 0; d < head_dim; ++d) {
                        cache_concat[b][i][offset + d] += w * cache_V[b][j][offset + d];
                    }
                }
            }
        }

        // Apply output projection: output = concat @ Wo
        std::vector<std::vector<float>> output_seq;
        output_seq.reserve(seq_len);

        for (std::size_t t = 0; t < seq_len; ++t) {
            std::vector<float> out_vec(embed_dim, 0.0f);
            for (int j = 0; j < embed_dim; ++j) {
                float val = 0.0f;
                for (int i = 0; i < embed_dim; ++i) {
                    val += cache_concat[b][t][i] * Wo[i][j];
                }
                out_vec[j] = val;
            }
            output_seq.push_back(std::move(out_vec));
        }

        output.push_back(std::move(output_seq));
    }

    return output;
}

// Resets all gradient matrices to zero
void MultiHeadAttention::zero_grad() {
    for (int i = 0; i < embed_dim; ++i) {
        for (int j = 0; j < embed_dim; ++j) {
            grad_Wq[i][j] = 0.0f;
            grad_Wk[i][j] = 0.0f;
            grad_Wv[i][j] = 0.0f;
            grad_Wo[i][j] = 0.0f;
        }
    }
}

// Computes gradients for Wq, Wk, Wv, Wo and returns gradient w.r.t. input
std::vector<std::vector<std::vector<float>>> MultiHeadAttention::backward(
    const std::vector<std::vector<std::vector<float>>>& input,
    const std::vector<std::vector<std::vector<float>>>& grad_output
) {
    if (input.size() != grad_output.size()) {
        throw std::runtime_error("MultiHeadAttention::backward: batch sizes must match");
    }

    std::size_t batch_size = input.size();
    std::vector<std::vector<std::vector<float>>> grad_input(
        batch_size,
        std::vector<std::vector<float>>(
            input[0].size(),
            std::vector<float>(embed_dim, 0.0f)
        )
    );

    for (std::size_t b = 0; b < batch_size; ++b) {
        std::size_t seq_len = input[b].size();

        if (grad_output[b].size() != seq_len) {
            throw std::runtime_error("MultiHeadAttention::backward: sequence lengths must match");
        }

        // Step 1: Compute grad_Wo and grad_concat
        // grad_Wo[i][j] = sum_t concat[b][t][i] * grad_output[b][t][j]
        std::vector<std::vector<float>> grad_concat(seq_len, std::vector<float>(embed_dim, 0.0f));

        for (std::size_t t = 0; t < seq_len; ++t) {
            for (int j = 0; j < embed_dim; ++j) {
                for (int i = 0; i < embed_dim; ++i) {
                    grad_Wo[i][j] += cache_concat[b][t][i] * grad_output[b][t][j];
                    grad_concat[t][i] += grad_output[b][t][j] * Wo[i][j];
                }
            }
        }

        // Step 2: For each head, run single-head backward
        std::vector<std::vector<float>> grad_Q(seq_len, std::vector<float>(embed_dim, 0.0f));
        std::vector<std::vector<float>> grad_K(seq_len, std::vector<float>(embed_dim, 0.0f));
        std::vector<std::vector<float>> grad_V(seq_len, std::vector<float>(embed_dim, 0.0f));

        for (int h = 0; h < num_heads; ++h) {
            int offset = h * head_dim;

            // Extract Q, K, V for this head
            std::vector<std::vector<float>> Q_head(seq_len, std::vector<float>(head_dim));
            std::vector<std::vector<float>> K_head(seq_len, std::vector<float>(head_dim));
            std::vector<std::vector<float>> V_head(seq_len, std::vector<float>(head_dim));
            std::vector<std::vector<float>> grad_head(seq_len, std::vector<float>(head_dim));

            for (std::size_t t = 0; t < seq_len; ++t) {
                for (int d = 0; d < head_dim; ++d) {
                    Q_head[t][d] = cache_Q[b][t][offset + d];
                    K_head[t][d] = cache_K[b][t][offset + d];
                    V_head[t][d] = cache_V[b][t][offset + d];
                    grad_head[t][d] = grad_concat[t][offset + d];
                }
            }

            // Run backward for this head
            auto [grad_Q_h, grad_K_h, grad_V_h] = head_backward(
                seq_len, head_dim, scale,
                Q_head, K_head, V_head,
                cache_weights[b][h], grad_head
            );

            // Place head gradients back into full gradient tensors
            for (std::size_t t = 0; t < seq_len; ++t) {
                for (int d = 0; d < head_dim; ++d) {
                    grad_Q[t][offset + d] = grad_Q_h[t][d];
                    grad_K[t][offset + d] = grad_K_h[t][d];
                    grad_V[t][offset + d] = grad_V_h[t][d];
                }
            }
        }

        // Step 3: Accumulate gradients for Wq, Wk, Wv
        for (std::size_t t = 0; t < seq_len; ++t) {
            for (int i = 0; i < embed_dim; ++i) {
                for (int j = 0; j < embed_dim; ++j) {
                    grad_Wq[i][j] += input[b][t][i] * grad_Q[t][j];
                    grad_Wk[i][j] += input[b][t][i] * grad_K[t][j];
                    grad_Wv[i][j] += input[b][t][i] * grad_V[t][j];
                }
            }
        }

        // Step 4: Compute grad_input = grad_Q @ Wq^T + grad_K @ Wk^T + grad_V @ Wv^T
        for (std::size_t t = 0; t < seq_len; ++t) {
            for (int i = 0; i < embed_dim; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < embed_dim; ++j) {
                    sum += grad_Q[t][j] * Wq[i][j];
                    sum += grad_K[t][j] * Wk[i][j];
                    sum += grad_V[t][j] * Wv[i][j];
                }
                grad_input[b][t][i] += sum;
            }
        }
    }

    return grad_input;
}

// Updates Wq, Wk, Wv, Wo using SGD
void MultiHeadAttention::apply_gradients(float learning_rate) {
    // Gradient clipping (per-layer L2 norm)
    const float max_norm = 1.0f;
    float grad_norm_sq = 0.0f;
    for (int i = 0; i < embed_dim; ++i) {
        for (int j = 0; j < embed_dim; ++j) {
            grad_norm_sq += grad_Wq[i][j] * grad_Wq[i][j];
            grad_norm_sq += grad_Wk[i][j] * grad_Wk[i][j];
            grad_norm_sq += grad_Wv[i][j] * grad_Wv[i][j];
            grad_norm_sq += grad_Wo[i][j] * grad_Wo[i][j];
        }
    }
    float grad_norm = std::sqrt(grad_norm_sq);
    float scale = (grad_norm > max_norm) ? (max_norm / grad_norm) : 1.0f;

    for (int i = 0; i < embed_dim; ++i) {
        for (int j = 0; j < embed_dim; ++j) {
            Wq[i][j] -= learning_rate * grad_Wq[i][j] * scale;
            Wk[i][j] -= learning_rate * grad_Wk[i][j] * scale;
            Wv[i][j] -= learning_rate * grad_Wv[i][j] * scale;
            Wo[i][j] -= learning_rate * grad_Wo[i][j] * scale;
        }
    }
}

void MultiHeadAttention::save_weights(std::ostream& f) const {
    for (int i = 0; i < embed_dim; ++i) {
        f.write(reinterpret_cast<const char*>(Wq[i].data()), embed_dim * sizeof(float));
        f.write(reinterpret_cast<const char*>(Wk[i].data()), embed_dim * sizeof(float));
        f.write(reinterpret_cast<const char*>(Wv[i].data()), embed_dim * sizeof(float));
        f.write(reinterpret_cast<const char*>(Wo[i].data()), embed_dim * sizeof(float));
    }
}

void MultiHeadAttention::load_weights(std::ifstream& f) {
    for (int i = 0; i < embed_dim; ++i) {
        f.read(reinterpret_cast<char*>(Wq[i].data()), embed_dim * sizeof(float));
        f.read(reinterpret_cast<char*>(Wk[i].data()), embed_dim * sizeof(float));
        f.read(reinterpret_cast<char*>(Wv[i].data()), embed_dim * sizeof(float));
        f.read(reinterpret_cast<char*>(Wo[i].data()), embed_dim * sizeof(float));
    }
}
