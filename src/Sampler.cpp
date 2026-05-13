#include "Sampler.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

// Always chooses the token with the biggest score
int Sampler::greedy(const std::vector<float>& logits) {
    if (logits.empty()) {
        throw std::runtime_error("Sampler::greedy requires non-empty logits");
    }

    int best_idx = 0;
    float best_val = logits[0];

    for (std::size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = static_cast<int>(i);
        }
    }

    return best_idx;
}

// Convenience wrapper with default top_k
int Sampler::sample(const std::vector<float>& logits, float temperature) {
    return sample(logits, temperature, 40, nullptr, 1.0f);
}

// Full sampling with temperature, top-k, and repetition penalty
int Sampler::sample(
    const std::vector<float>& logits,
    float temperature,
    int top_k,
    const std::vector<int>* recent_tokens,
    float repetition_penalty
) {
    if (logits.empty()) {
        throw std::runtime_error("Sampler::sample requires non-empty logits");
    }

    // Greedy decoding when temperature is zero
    if (temperature <= 0.0f) {
        return greedy(logits);
    }

    // Work on a copy so we can modify
    std::vector<float> adjusted_logits = logits;

    // Apply repetition penalty: divide logits of recently used tokens
    if (recent_tokens != nullptr && repetition_penalty > 1.0f) {
        for (int token_id : *recent_tokens) {
            if (token_id >= 0 && token_id < static_cast<int>(adjusted_logits.size())) {
                if (adjusted_logits[token_id] > 0.0f) {
                    adjusted_logits[token_id] /= repetition_penalty;
                } else {
                    adjusted_logits[token_id] *= repetition_penalty;
                }
            }
        }
    }

    // Scale by temperature
    for (float& logit : adjusted_logits) {
        logit /= temperature;
    }

    // Find top-k tokens
    std::vector<std::pair<float, int>> indexed_logits;
    indexed_logits.reserve(adjusted_logits.size());
    for (std::size_t i = 0; i < adjusted_logits.size(); ++i) {
        indexed_logits.emplace_back(adjusted_logits[i], static_cast<int>(i));
    }

    // Partial sort to find top k
    std::size_t k = static_cast<std::size_t>(top_k);
    if (k < indexed_logits.size()) {
        std::nth_element(
            indexed_logits.begin(),
            indexed_logits.begin() + k,
            indexed_logits.end(),
            std::greater<std::pair<float, int>>()
        );
        indexed_logits.resize(k);
    }

    // Softmax over top-k only
    float max_logit = indexed_logits[0].first;
    for (const auto& p : indexed_logits) {
        if (p.first > max_logit) {
            max_logit = p.first;
        }
    }

    float sum_exp = 0.0f;
    for (auto& p : indexed_logits) {
        p.first = std::exp(p.first - max_logit);
        sum_exp += p.first;
    }

    for (auto& p : indexed_logits) {
        p.first /= sum_exp;
    }

    // Sort by probability for cumulative sampling
    std::sort(indexed_logits.begin(), indexed_logits.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Sample from cumulative distribution
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);

    float cumulative = 0.0f;
    for (const auto& p : indexed_logits) {
        cumulative += p.first;
        if (r <= cumulative) {
            return p.second;
        }
    }

    // Fallback to highest probability
    return indexed_logits.back().second;
}
