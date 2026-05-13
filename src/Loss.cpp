#include "Loss.hpp"

#include <cmath>
#include <stdexcept>

// Converts logits to probabilities with numerical stability (subtracts max before exp)
std::vector<float> Loss::softmax(const std::vector<float>& logits) {
    if (logits.empty()) {
        throw std::runtime_error("softmax requires at least one logit");
    }

    float max_logit = logits[0];
    for (float logit : logits) {
        if (logit > max_logit) {
            max_logit = logit;
        }
    }

    std::vector<float> probabilities(logits.size());
    float sum_exp = 0.0f;

    for (std::size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum_exp += probabilities[i];
    }

    for (float& probability : probabilities) {
        probability /= sum_exp;
    }

    return probabilities;
}

// Computes numerically stable cross-entropy averaged over all tokens
float Loss::cross_entropy(
    const std::vector<std::vector<std::vector<float>>>& logits,
    const std::vector<std::vector<int>>& targets
) {
    if (logits.empty()) {
        throw std::runtime_error("cross_entropy requires non-empty logits");
    }

    if (logits.size() != targets.size()) {
        throw std::runtime_error("Logits batch size must match targets batch size");
    }

    double total_loss = 0.0;
    std::size_t count = 0;

    for (std::size_t batch = 0; batch < logits.size(); ++batch) {
        if (logits[batch].size() != targets[batch].size()) {
            throw std::runtime_error("Logits sequence length must match targets sequence length");
        }

        for (std::size_t position = 0; position < logits[batch].size(); ++position) {
            const std::vector<float>& token_logits = logits[batch][position];

            if (token_logits.empty()) {
                throw std::runtime_error("Each logits vector must contain at least one score");
            }

            int target_id = targets[batch][position];
            if (target_id < 0 || target_id >= static_cast<int>(token_logits.size())) {
                throw std::runtime_error("Target token ID out of bounds in cross_entropy");
            }

            float max_logit = token_logits[0];
            for (float logit : token_logits) {
                if (logit > max_logit) {
                    max_logit = logit;
                }
            }

            double sum_exp = 0.0;
            for (float logit : token_logits) {
                sum_exp += std::exp(static_cast<double>(logit - max_logit));
            }

            double log_sum_exp = static_cast<double>(max_logit) + std::log(sum_exp);
            total_loss += log_sum_exp - static_cast<double>(token_logits[target_id]);
            ++count;
        }
    }

    if (count == 0) {
        throw std::runtime_error("cross_entropy requires at least one target");
    }

    return static_cast<float>(total_loss / static_cast<double>(count));
}

// Computes gradient: (softmax(logits) - one_hot(target)) / (batch_size * seq_len)
std::vector<std::vector<std::vector<float>>> Loss::backward(
    const std::vector<std::vector<std::vector<float>>>& logits,
    const std::vector<std::vector<int>>& targets
) {
    if (logits.empty()) {
        throw std::runtime_error("backward requires non-empty logits");
    }

    if (logits.size() != targets.size()) {
        throw std::runtime_error("Logits batch size must match targets batch size in backward");
    }

    std::size_t total_count = 0;
    for (const auto& batch : logits) {
        total_count += batch.size();
    }

    if (total_count == 0) {
        throw std::runtime_error("backward requires at least one target");
    }

    float scale = 1.0f / static_cast<float>(total_count);

    std::vector<std::vector<std::vector<float>>> grad;
    grad.reserve(logits.size());

    for (std::size_t b = 0; b < logits.size(); ++b) {
        if (logits[b].size() != targets[b].size()) {
            throw std::runtime_error("Logits sequence length must match targets sequence length in backward");
        }

        grad.emplace_back();
        grad[b].reserve(logits[b].size());

        for (std::size_t t = 0; t < logits[b].size(); ++t) {
            std::vector<float> probs = softmax(logits[b][t]);
            int target_id = targets[b][t];

            if (target_id < 0 || target_id >= static_cast<int>(probs.size())) {
                throw std::runtime_error("Target token ID out of bounds in backward");
            }

            probs[target_id] -= 1.0f;

            for (float& p : probs) {
                p *= scale;
            }

            grad[b].push_back(std::move(probs));
        }
    }

    return grad;
}
