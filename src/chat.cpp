#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "ByteTokenizer.hpp"
#include "GPT.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.bin>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];

    // Model config (must match training config)
    const int vocab_size = 257;
    const int context_length = 64;
    const int embed_dim = 128;
    const int num_heads = 4;
    const int num_layers = 2;

    ByteTokenizer tokenizer;

    std::cout << "==========================" << std::endl;
    std::cout << "Darija GPT Chat" << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "Loading model from: " << model_path << std::endl;

    GPT model(vocab_size, context_length, embed_dim, num_heads, num_layers);

    if (!model.load(model_path)) {
        std::cerr << "Failed to load model from " << model_path << std::endl;
        return 1;
    }

    std::cout << "Model loaded!" << std::endl;
    std::cout << "\nCommands:" << std::endl;
    std::cout << "  /quit       - exit chat" << std::endl;
    std::cout << "  /temp N     - set temperature (default 0.8)" << std::endl;
    std::cout << "  /len N      - set max response length (default 100)" << std::endl;
    std::cout << "  Type anything else to chat!" << std::endl;
    std::cout << "==========================" << std::endl;

    float temperature = 0.8f;
    int max_length = 100;
    std::vector<int> conversation_history;

    while (true) {
        std::cout << "\nYou: ";
        std::string input;
        std::getline(std::cin, input);

        if (input == "/quit") {
            std::cout << "Goodbye!" << std::endl;
            break;
        }

        if (input.rfind("/temp ", 0) == 0) {
            try {
                temperature = std::stof(input.substr(6));
                std::cout << "Temperature set to " << temperature << std::endl;
            } catch (...) {
                std::cerr << "Invalid temperature value." << std::endl;
            }
            continue;
        }

        if (input.rfind("/len ", 0) == 0) {
            try {
                max_length = std::stoi(input.substr(5));
                std::cout << "Max length set to " << max_length << std::endl;
            } catch (...) {
                std::cerr << "Invalid length value." << std::endl;
            }
            continue;
        }

        // Encode user input and add to history
        auto user_tokens = tokenizer.encode(input + "\n");
        conversation_history.insert(conversation_history.end(),
                                    user_tokens.begin(), user_tokens.end());

        // Truncate to context length
        if (static_cast<int>(conversation_history.size()) > context_length) {
            conversation_history = std::vector<int>(
                conversation_history.end() - context_length,
                conversation_history.end()
            );
        }

        // Generate response
        auto response_tokens = model.generate(conversation_history, max_length, temperature);

        // Extract only the new tokens (after the prompt)
        std::vector<int> new_tokens;
        if (response_tokens.size() > conversation_history.size()) {
            new_tokens.assign(
                response_tokens.begin() + conversation_history.size(),
                response_tokens.end()
            );
        }

        std::string response = tokenizer.decode(new_tokens);
        std::cout << "Bot: " << response << std::endl;

        // Add response to history
        conversation_history.insert(conversation_history.end(),
                                    new_tokens.begin(), new_tokens.end());
    }

    return 0;
}
