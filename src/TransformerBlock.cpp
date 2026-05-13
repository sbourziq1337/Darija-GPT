#include "TransformerBlock.hpp"

#include <stdexcept>

TransformerBlock::TransformerBlock(int embed_dim, int num_heads)
    : ln1(embed_dim),
      attn(embed_dim, num_heads),
      ln2(embed_dim),
      ff(embed_dim) {
}

std::vector<std::vector<std::vector<float>>> TransformerBlock::forward(
    const std::vector<std::vector<std::vector<float>>>& input
) {
    cache_input = input;

    // First sub-block: LayerNorm -> Attention -> Residual
    cache_ln1_out = ln1.forward(input);
    auto attn_out = attn.forward(cache_ln1_out);

    // Residual add: x + Attention(LayerNorm(x))
    cache_after_attn = input;
    for (std::size_t b = 0; b < input.size(); ++b) {
        for (std::size_t t = 0; t < input[b].size(); ++t) {
            for (std::size_t d = 0; d < input[b][t].size(); ++d) {
                cache_after_attn[b][t][d] += attn_out[b][t][d];
            }
        }
    }

    // Second sub-block: LayerNorm -> FeedForward -> Residual
    cache_ln2_out = ln2.forward(cache_after_attn);
    auto ff_out = ff.forward(cache_ln2_out);

    // Residual add: x + FeedForward(LayerNorm(x))
    std::vector<std::vector<std::vector<float>>> output = cache_after_attn;
    for (std::size_t b = 0; b < cache_after_attn.size(); ++b) {
        for (std::size_t t = 0; t < cache_after_attn[b].size(); ++t) {
            for (std::size_t d = 0; d < cache_after_attn[b][t].size(); ++d) {
                output[b][t][d] += ff_out[b][t][d];
            }
        }
    }

    return output;
}

void TransformerBlock::zero_grad() {
    ln1.zero_grad();
    attn.zero_grad();
    ln2.zero_grad();
    ff.zero_grad();
}

std::vector<std::vector<std::vector<float>>> TransformerBlock::backward(
    const std::vector<std::vector<std::vector<float>>>& grad_output
) {
    if (grad_output.empty()) {
        throw std::runtime_error("TransformerBlock::backward: grad_output is empty");
    }

    // Backprop through second residual: output = after_attn + ff_out
    // grad_after_attn = grad_output (residual path)
    // grad_ff_out = grad_output (feedforward path)
    auto grad_after_attn = grad_output;
    auto grad_ff_out = grad_output;

    // Backprop through ff: ff took ln2_out as input
    auto grad_ln2_out = ff.backward(cache_ln2_out, grad_ff_out);

    // Backprop through ln2 (this gives gradient w.r.t. pre-LN space)
    auto grad_after_attn_from_ln2 = ln2.backward(grad_ln2_out);

    // Add ln2 path to grad_after_attn
    for (std::size_t b = 0; b < grad_after_attn.size(); ++b) {
        for (std::size_t t = 0; t < grad_after_attn[b].size(); ++t) {
            for (std::size_t d = 0; d < grad_after_attn[b][t].size(); ++d) {
                grad_after_attn[b][t][d] += grad_after_attn_from_ln2[b][t][d];
            }
        }
    }

    // Backprop through first residual: after_attn = input + attn_out
    // grad_input = grad_after_attn (residual path)
    // grad_attn_out = grad_after_attn (attention path)
    auto grad_input = grad_after_attn;
    auto grad_attn_out = grad_after_attn;

    // Backprop through attn: attn took ln1_out as input
    auto grad_ln1_out = attn.backward(cache_ln1_out, grad_attn_out);

    // Backprop through ln1 (this gives gradient w.r.t. pre-LN space)
    auto grad_input_from_ln1 = ln1.backward(grad_ln1_out);

    // Add ln1 path to grad_input
    for (std::size_t b = 0; b < grad_input.size(); ++b) {
        for (std::size_t t = 0; t < grad_input[b].size(); ++t) {
            for (std::size_t d = 0; d < grad_input[b][t].size(); ++d) {
                grad_input[b][t][d] += grad_input_from_ln1[b][t][d];
            }
        }
    }

    return grad_input;
}

void TransformerBlock::apply_gradients(float learning_rate) {
    ln1.apply_gradients(learning_rate);
    attn.apply_gradients(learning_rate);
    ln2.apply_gradients(learning_rate);
    ff.apply_gradients(learning_rate);
}

void TransformerBlock::save_weights(std::ofstream& f) const {
    ln1.save_weights(f);
    attn.save_weights(f);
    ln2.save_weights(f);
    ff.save_weights(f);
}

void TransformerBlock::load_weights(std::ifstream& f) {
    ln1.load_weights(f);
    attn.load_weights(f);
    ln2.load_weights(f);
    ff.load_weights(f);
}
