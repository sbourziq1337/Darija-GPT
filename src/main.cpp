#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ByteTokenizer.hpp"
#include "Dataset.hpp"
#include "GPT.hpp"
#include "Trainer.hpp"

int main() {
    std::cout << "==========================" << std::endl;
    std::cout << "Darija GPT - Full Model" << std::endl;
    std::cout << "==========================" << std::endl;

    // Model config
    const int vocab_size = 257;
    const int context_length = 64;
    const int embed_dim = 128;
    const int num_heads = 4;
    const int num_layers = 2;
    const int batch_size = 8;
    const float learning_rate = 0.01f;
    const float lr_decay = 0.995f;
    const int decay_every = 200;
    const int training_steps = 10000;
    const int print_every = 250;

    std::cout << "\n--- Model Config ---" << std::endl;
    std::cout << "Vocab size:      " << vocab_size << std::endl;
    std::cout << "Context length:  " << context_length << std::endl;
    std::cout << "Embedding dim:   " << embed_dim << std::endl;
    std::cout << "Num heads:       " << num_heads << std::endl;
    std::cout << "Num layers:      " << num_layers << std::endl;
    std::cout << "Batch size:      " << batch_size << std::endl;
    std::cout << "Learning rate:   " << learning_rate << std::endl;
    std::cout << "Training steps:  " << training_steps << std::endl;

    // Load data
    std::cout << "\n[1/7] Loading data..." << std::endl;
    std::ifstream file("data/darija.txt");
    if (!file) {
        std::cerr << "Error: Could not open data file." << std::endl;
        return 1;
    }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    std::cout << "Data loaded: " << text.size() << " bytes." << std::endl;

    // Tokenize
    std::cout << "\n[2/7] Tokenizing..." << std::endl;
    ByteTokenizer tokenizer;
    std::vector<int> tokens = tokenizer.encode_with_eot(text);
    std::cout << "Tokens: " << tokens.size() << std::endl;

    // Dataset
    std::cout << "\n[3/7] Creating dataset..." << std::endl;
    Dataset dataset(tokens, batch_size, context_length);
    std::cout << "Dataset ready." << std::endl;

    // Create model
    std::cout << "\n[4/7] Creating GPT model..." << std::endl;
    GPT model(vocab_size, context_length, embed_dim, num_heads, num_layers);
    std::cout << "Model created." << std::endl;

    // Check for saved model
    std::string model_path = "models/darija_gpt.bin";
    if (std::filesystem::exists(model_path)) {
        std::cout << "\nFound saved model at " << model_path << std::endl;
        std::cout << "Loading weights..." << std::endl;
        if (model.load(model_path)) {
            std::cout << "Model loaded successfully!" << std::endl;
        } else {
            std::cout << "Failed to load model, training from scratch." << std::endl;
        }
    }

    // Train
    std::cout << "\n[5/7] Training..." << std::endl;
    Trainer trainer(dataset, model, learning_rate, lr_decay, decay_every, print_every);
    trainer.train(training_steps);

    // Evaluate
    std::cout << "\n[6/7] Evaluating..." << std::endl;
    float train_loss = trainer.evaluate(Split::Train, 20);
    float val_loss = trainer.evaluate(Split::Validation, 20);
    std::cout << "Train loss: " << train_loss << std::endl;
    std::cout << "Val loss:   " << val_loss << std::endl;
    std::cout << "Random baseline: log(" << vocab_size << ") = " << std::log(vocab_size) << std::endl;

    // Save model
    std::cout << "\n[7/7] Saving model to " << model_path << "..." << std::endl;
    if (model.save(model_path)) {
        std::cout << "Model saved!" << std::endl;
    } else {
        std::cerr << "Failed to save model." << std::endl;
    }

    // Generate samples
    std::cout << "\n--- Generation Samples ---" << std::endl;

    std::vector<std::string> prompts = {"salam ", "kifach ", "wa7ed "};

    for (const auto& prompt : prompts) {
        auto prompt_tokens = tokenizer.encode(prompt);
        auto generated = model.generate(prompt_tokens, 100, 0.8f);
        std::string text = tokenizer.decode(generated);

        std::cout << "\nPrompt: \"" << prompt << "\"" << std::endl;
        std::cout << "Generated: " << text << std::endl;
    }

    return 0;
}
