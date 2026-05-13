#include "GPT.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "Checkpoint.hpp"
#include "Sampler.hpp"

namespace {
    void write_int(std::ostream& f, int value) {
        f.write(reinterpret_cast<const char*>(&value), sizeof(int));
    }

    int read_int(std::ifstream& f) {
        int value = 0;
        f.read(reinterpret_cast<char*>(&value), sizeof(int));
        return value;
    }

    void write_float(std::ostream& f, float value) {
        f.write(reinterpret_cast<const char*>(&value), sizeof(float));
    }

    float read_float(std::ifstream& f) {
        float value = 0.0f;
        f.read(reinterpret_cast<char*>(&value), sizeof(float));
        return value;
    }

    void write_uint32(std::ostream& f, uint32_t value) {
        f.write(reinterpret_cast<const char*>(&value), sizeof(uint32_t));
    }

    uint32_t read_uint32(std::ifstream& f) {
        uint32_t value = 0;
        f.read(reinterpret_cast<char*>(&value), sizeof(uint32_t));
        return value;
    }
}

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

bool GPT::save(const std::string& path) const {
    CheckpointMetadata meta;
    return save(path, meta);
}

bool GPT::save(const std::string& path, const CheckpointMetadata& meta) const {
    std::filesystem::path dir = std::filesystem::path(path).parent_path();
    if (!dir.empty() && !std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }

    // Build the checkpoint in memory first
    std::ostringstream oss(std::ios::binary);
    std::ostream& f = oss;

    // Header
    f.write(CheckpointMetadata::MAGIC, 8);
    write_int(f, CheckpointMetadata::CURRENT_VERSION);
    write_int(f, vocab_size_);
    write_int(f, context_length_);
    write_int(f, embed_dim_);
    write_int(f, num_heads_);
    write_int(f, num_layers_);

    // Metadata (version 2)
    write_int(f, meta.saved_step);
    write_float(f, meta.saved_lr);
    write_float(f, meta.best_loss);
    write_uint32(f, meta.rng_seed);
    write_uint32(f, 0);  // checksum placeholder (computed later)

    // Weights
    embedding_.save_weights(f);
    for (const auto& block : blocks_) {
        block.save_weights(f);
    }
    final_ln_.save_weights(f);
    lm_head_.save_weights(f);

    std::string data = oss.str();

    // Compute checksum over everything AFTER the checksum field
    // Skip magic(8) + version(4) + config(20) + meta_step(4) + meta_lr(4) + meta_loss(4) + meta_seed(4) = 48
    // Checksum field itself is at offset 48, 4 bytes
    std::size_t checksum_offset = 8 + 4 + 20 + 4 + 4 + 4 + 4;
    uint32_t checksum = CheckpointUtil::compute_checksum(
        reinterpret_cast<const float*>(data.data() + checksum_offset + 4),
        (data.size() - checksum_offset - 4) / sizeof(float)
    );

    // Write checksum into the buffer
    *reinterpret_cast<uint32_t*>(data.data() + checksum_offset) = checksum;

    // Atomic save
    return CheckpointUtil::atomic_save(path, ".tmp", data.data(), data.size());
}

bool GPT::load(const std::string& path) {
    CheckpointMetadata dummy;
    return load(path, dummy);
}

bool GPT::load(const std::string& path, CheckpointMetadata& out_meta) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }

    char magic[9] = {};
    f.read(magic, 8);
    if (std::string(magic, 8) != CheckpointMetadata::MAGIC) {
        std::cerr << "GPT::load: invalid magic header" << std::endl;
        return false;
    }

    int version = read_int(f);
    if (version != 1 && version != CheckpointMetadata::CURRENT_VERSION) {
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

    if (version == CheckpointMetadata::CURRENT_VERSION) {
        // Read metadata
        out_meta.version = version;
        out_meta.saved_step = read_int(f);
        out_meta.saved_lr = read_float(f);
        out_meta.best_loss = read_float(f);
        out_meta.rng_seed = read_uint32(f);
        out_meta.checksum = read_uint32(f);

        // Validate checksum
        // Get current position
        auto pos = f.tellg();
        f.seekg(0, std::ios::end);
        auto end = f.tellg();
        std::size_t data_size = static_cast<std::size_t>(end - pos);
        f.seekg(pos);

        std::vector<float> weight_data(data_size / sizeof(float));
        f.read(reinterpret_cast<char*>(weight_data.data()), static_cast<std::streamsize>(data_size));

        uint32_t computed = CheckpointUtil::compute_checksum(weight_data.data(), weight_data.size());
        if (computed != out_meta.checksum) {
            std::cerr << "GPT::load: checksum mismatch! File may be corrupted." << std::endl;
            std::cerr << "  Expected: " << out_meta.checksum << std::endl;
            std::cerr << "  Computed: " << computed << std::endl;
            return false;
        }

        // Rewind to load weights
        f.seekg(pos);
    } else {
        // Version 1: no metadata
        out_meta = CheckpointMetadata();
    }

    embedding_.load_weights(f);
    for (auto& block : blocks_) {
        block.load_weights(f);
    }
    final_ln_.load_weights(f);
    lm_head_.load_weights(f);

    return f.good();
}
