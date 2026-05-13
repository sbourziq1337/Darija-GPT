#pragma once

#include <vector>

// Loss functions: softmax and cross-entropy for language model training
class Loss {
public:
    // Converts logits to probabilities using softmax
    static std::vector<float> softmax(const std::vector<float>& logits);

    // Computes average cross-entropy loss between predictions and targets
    static float cross_entropy(
        const std::vector<std::vector<std::vector<float>>>& logits,
        const std::vector<std::vector<int>>& targets
    );

    // Computes gradient of loss w.r.t. logits: (softmax - one_hot) / count
    static std::vector<std::vector<std::vector<float>>> backward(
        const std::vector<std::vector<std::vector<float>>>& logits,
        const std::vector<std::vector<int>>& targets
    );
};
