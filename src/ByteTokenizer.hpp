#pragma once

#include <string>
#include <vector>

// Byte-level tokenizer: converts text to token IDs (0-255) and back. EOT token = 256.
class ByteTokenizer {
public:
    static constexpr int EOT_TOKEN = 256;
    static constexpr int VOCAB_SIZE = EOT_TOKEN + 1;

    ByteTokenizer() = default;

    // Returns vocabulary size (257)
    int vocab_size() const;

    // Converts text string to list of token IDs
    std::vector<int> encode(const std::string& text) const;

    // Converts list of token IDs back to text string
    std::string decode(const std::vector<int>& tokens) const;

    // Encodes text and appends EOT token (256) at the end
    std::vector<int> encode_with_eot(const std::string& text) const;
};
