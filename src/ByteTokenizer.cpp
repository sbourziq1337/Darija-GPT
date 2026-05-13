#include "ByteTokenizer.hpp"

// Returns total number of tokens (257: bytes 0-255 + EOT)
int ByteTokenizer::vocab_size() const {
    return VOCAB_SIZE;
}

// Converts each character to its raw byte value
std::vector<int> ByteTokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    ids.reserve(text.size());

    for (unsigned char c : text) {
        ids.push_back(static_cast<int>(c));
    }

    return ids;
}

// Converts token IDs back to characters (skips EOT token 256)
std::string ByteTokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;
    text.reserve(tokens.size());

    for (int id : tokens) {
        if (id >= 0 && id <= 255) {
            text.push_back(static_cast<char>(id));
        }
    }

    return text;
}

// Encodes text and adds end-of-text token (256) at the end
std::vector<int> ByteTokenizer::encode_with_eot(const std::string& text) const {
    std::vector<int> ids = encode(text);
    ids.push_back(EOT_TOKEN);
    return ids;
}
