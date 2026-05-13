#pragma once

#include <cstdint>
#include <string>

// Checkpoint metadata stored in model binary files (format version 2)
struct CheckpointMetadata {
    int version = 2;           // File format version
    int saved_step = 0;        // Training step when saved
    float saved_lr = 0.0f;     // Learning rate when saved
    float best_loss = 9999.0f; // Best loss seen so far
    uint32_t checksum = 0;     // Simple checksum for corruption detection
    uint32_t rng_seed = 0;     // Random number generator seed

    static constexpr int CURRENT_VERSION = 2;
    static constexpr const char* MAGIC = "DARIJGPT";
};

// Checkpoint utility functions
namespace CheckpointUtil {
    // Compute simple checksum (XOR-based) for a float buffer
    uint32_t compute_checksum(const float* data, std::size_t count);

    // Atomic save: write to temp path, then rename
    bool atomic_save(const std::string& final_path,
                     const std::string& temp_suffix,
                     const char* data, std::size_t size);

    // Validate that a checkpoint file exists and is readable
    bool validate_file(const std::string& path);
}
