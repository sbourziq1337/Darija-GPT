#pragma once

#include <vector>

// Improved sampler with top-k filtering and repetition penalty
class Sampler {
public:
    // Greedy: always pick highest probability token
    static int greedy(const std::vector<float>& logits);

    // Sample with temperature, top-k filtering, and optional repetition penalty
    static int sample(
        const std::vector<float>& logits,
        float temperature,
        int top_k = 40,
        const std::vector<int>* recent_tokens = nullptr,
        float repetition_penalty = 1.0f
    );

    // Convenience: sample with only temperature (uses default top_k=40)
    static int sample(const std::vector<float>& logits, float temperature);
};
