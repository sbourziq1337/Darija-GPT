#include "Checkpoint.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>

uint32_t CheckpointUtil::compute_checksum(const float* data, std::size_t count) {
    uint32_t checksum = 0xDEADBEEF;
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
    std::size_t byte_count = count * sizeof(float);
    for (std::size_t i = 0; i < byte_count; ++i) {
        checksum ^= static_cast<uint32_t>(bytes[i]) << ((i % 4) * 8);
        checksum = (checksum << 1) | (checksum >> 31);  // Rotate left
    }
    return checksum;
}

bool CheckpointUtil::atomic_save(const std::string& final_path,
                                  const std::string& temp_suffix,
                                  const char* data, std::size_t size) {
    std::string temp_path = final_path + temp_suffix;

    std::ofstream f(temp_path, std::ios::binary);
    if (!f) {
        std::cerr << "CheckpointUtil::atomic_save: cannot open temp file " << temp_path << std::endl;
        return false;
    }

    f.write(data, static_cast<std::streamsize>(size));
    if (!f.good()) {
        std::cerr << "CheckpointUtil::atomic_save: write failed for " << temp_path << std::endl;
        return false;
    }
    f.close();

    // Atomic rename
    if (std::rename(temp_path.c_str(), final_path.c_str()) != 0) {
        std::cerr << "CheckpointUtil::atomic_save: rename failed from " << temp_path
                  << " to " << final_path << std::endl;
        return false;
    }

    return true;
}

bool CheckpointUtil::validate_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return false;
    }
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);

    // Minimum size: magic (8) + version (4) + config (20) + metadata rest (~24) = ~56 bytes
    if (size < 56) {
        std::cerr << "CheckpointUtil::validate_file: file too small (" << size << " bytes)" << std::endl;
        return false;
    }

    // Read and verify magic
    char magic[9] = {};
    f.read(magic, 8);
    if (std::string(magic, 8) != CheckpointMetadata::MAGIC) {
        std::cerr << "CheckpointUtil::validate_file: invalid magic header" << std::endl;
        return false;
    }

    // Read and verify version
    int version = 0;
    f.read(reinterpret_cast<char*>(&version), sizeof(int));
    if (version != CheckpointMetadata::CURRENT_VERSION) {
        std::cerr << "CheckpointUtil::validate_file: unsupported version " << version << std::endl;
        return false;
    }

    return true;
}
