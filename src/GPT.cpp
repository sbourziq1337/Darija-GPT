#include "GPT.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "Sampler.hpp"

GPT::GPT(int vocab_size, int context_length, int embed_dim, int num_heads, int num_layers)
    : vocab_size_(vocab_size),
      context_length_(context_length),
      embed_dim_(embed_dim),
      num_heads_(num_heads),
      num_layers_(num_layers),
      embedding_(vocab_size, embed_dim, context_length),
      final_ln_(embed_dim),
      lm_head_(embed_dim, vocab_size, 0.02f) {

    if (num_layers <= 0) {
        throw std::runtime_error("GPT: num_layers must be greater than zero");
    }

    blocks_.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        blocks_.emplace_back(embed_dim, num_heads);
    }
}

std::vector<std::vector<std::vector<float>>> GPT::forward(
    const std::vector<std::vector<int>>& input_batch
) {
    cache_input_ids_ = input_batch;

    // Embedding: [B][T] -> [B][T][E]
    cache_embedded_ = embedding_.forward(input_batch);

    auto x = cache_embedded_;

    // Transformer blocks
    for (auto& block : blocks_) {
        x = block.forward(x);
    }

    // Final layer norm
    cache_final_ln_ = final_ln_.forward(x);

    // Language modeling head: [B][T][E] -> [B][T][V]
    return lm_head_.forward(cache_final_ln_);
}

void GPT::backward(
    const std::vector<std::vector<int>>& input_batch,
    const std::vector<std::vector<std::vector<float>>>& grad_logits
) {
    // Backward through language head
    auto grad_final_ln = lm_head_.backward(cache_final_ln_, grad_logits);

    // Backward through final layer norm
    auto grad_blocks = final_ln_.backward(grad_final_ln);

    // Backward through transformer blocks (reverse order)
    for (int i = static_cast<int>(blocks_.size()) - 1; i >= 0; --i) {
        grad_blocks = blocks_[i].backward(grad_blocks);
    }

    // Backward through embedding
    embedding_.backward(input_batch, grad_blocks);
}

void GPT::zero_grad() {
    embedding_.zero_grad();
    for (auto& block : blocks_) {
        block.zero_grad();
    }
    final_ln_.zero_grad();
    lm_head_.zero_grad();
}

void GPT::apply_gradients(float learning_rate) {
    embedding_.apply_gradients(learning_rate);
    for (auto& block : blocks_) {
        block.apply_gradients(learning_rate);
    }
    final_ln_.apply_gradients(learning_rate);
    lm_head_.apply_gradients(learning_rate);
}

std::vector<int> GPT::generate(
    const std::vector<int>& prompt,
    int max_tokens,
    float temperature
) {
    std::vector<int> tokens = prompt;
    const int top_k = 40;
    const float repetition_penalty = 1.2f;
    const int repetition_window = 16;  // Penalize last 16 tokens

    for (int i = 0; i < max_tokens; ++i) {
        // Truncate to context_length
        std::vector<int> context = tokens;
        if (static_cast<int>(context.size()) > context_length_) {
            context = std::vector<int>(context.end() - context_length_, context.end());
        }

        // Forward pass
        std::vector<std::vector<int>> batch = {context};
        auto logits = forward(batch);

        if (logits.empty() || logits[0].empty()) {
            break;
        }

        // Get recent tokens for repetition penalty
        std::vector<int> recent_tokens;
        if (tokens.size() > static_cast<std::size_t>(repetition_window)) {
            recent_tokens.assign(tokens.end() - repetition_window, tokens.end());
        } else {
            recent_tokens = tokens;
        }

        // Sample next token with top-k and repetition penalty
        int next_token = Sampler::sample(
            logits[0].back(),
            temperature,
            top_k,
            &recent_tokens,
            repetition_penalty
        );

        if (next_token == ByteTokenizer::EOT_TOKEN) {
            break;
        }

        tokens.push_back(next_token);
    }

    return tokens;
}

// ---------------------------------------------------------------------------
// Save / Load helpers
// ---------------------------------------------------------------------------

static void write_int(std::ofstream& f, int value) {
    f.write(reinterpret_cast<const char*>(&value), sizeof(int));
}

static int read_int(std::ifstream& f) {
    int value = 0;
    f.read(reinterpret_cast<char*>(&value), sizeof(int));
    return value;
}

bool GPT::save(const std::string& path) const {
    std::filesystem::path dir = std::filesystem::path(path).parent_path();
    if (!dir.empty() && !std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "GPT::save: cannot open " << path << std::endl;
        return false;
    }

    // Header
    const char magic[9] = "DARIJGPT";
    f.write(magic, 8);
    write_int(f, 1);               // version
    write_int(f, vocab_size_);
    write_int(f, context_length_);
    write_int(f, embed_dim_);
    write_int(f, num_heads_);
    write_int(f, num_layers_);

    // Weights
    embedding_.save_weights(f);
    for (const auto& block : blocks_) {
        block.save_weights(f);
    }
    final_ln_.save_weights(f);
    lm_head_.save_weights(f);

    return f.good();
}

bool GPT::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }

    char magic[9] = {};
    f.read(magic, 8);
    if (std::string(magic, 8) != "DARIJGPT") {
        std::cerr << "GPT::load: invalid magic header" << std::endl;
        return false;
    }

    int version = read_int(f);
    if (version != 1) {
        std::cerr << "GPT::load: unsupported version " << version << std::endl;
        return false;
    }

    int vocab_size = read_int(f);
    int context_length = read_int(f);
    int embed_dim = read_int(f);
    int num_heads = read_int(f);
    int num_layers = read_int(f);

    if (vocab_size != vocab_size_ || context_length != context_length_ ||
        embed_dim != embed_dim_ || num_heads != num_heads_ || num_layers != num_layers_) {
        std::cerr << "GPT::load: config mismatch" << std::endl;
        std::cerr << "  File:  vocab=" << vocab_size
                  << " ctx=" << context_length
                  << " embed=" << embed_dim
                  << " heads=" << num_heads
                  << " layers=" << num_layers << std::endl;
        std::cerr << "  Model: vocab=" << vocab_size_
                  << " ctx=" << context_length_
                  << " embed=" << embed_dim_
                  << " heads=" << num_heads_
                  << " layers=" << num_layers_ << std::endl;
        return false;
    }

    embedding_.load_weights(f);
    for (auto& block : blocks_) {
        block.load_weights(f);
    }
    final_ln_.load_weights(f);
    lm_head_.load_weights(f);

    return f.good();
}
