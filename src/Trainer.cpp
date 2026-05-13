#include "Trainer.hpp"

#include <chrono>
#include <iostream>

#include "Loss.hpp"

Trainer::Trainer(Dataset& dataset, GPT& model, float learning_rate,
                 float lr_decay, int decay_every, int print_every)
    : dataset(dataset),
      model(model),
      learning_rate(learning_rate),
      lr_decay(lr_decay),
      decay_every(decay_every),
      print_every(print_every) {}

void Trainer::train(int steps) {
    float current_lr = learning_rate;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < steps; ++step) {
        auto step_start = std::chrono::high_resolution_clock::now();

        Batch batch = dataset.get_batch(Split::Train);

        model.zero_grad();

        auto logits = model.forward(batch.inputs);
        float loss = Loss::cross_entropy(logits, batch.targets);

        auto grad_logits = Loss::backward(logits, batch.targets);
        model.backward(batch.inputs, grad_logits);

        model.apply_gradients(current_lr);

        if ((step + 1) % print_every == 0 || step == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            float elapsed = std::chrono::duration<float>(now - start_time).count();
            float step_time = std::chrono::duration<float>(now - step_start).count();
            float est_remaining = (steps - (step + 1)) * (elapsed / (step + 1));

            std::cout << "Step " << (step + 1) << "/" << steps
                      << " | Loss: " << loss
                      << " | LR: " << current_lr
                      << " | " << step_time << "s/step"
                      << " | ~" << static_cast<int>(est_remaining / 60.0f) << "min left"
                      << std::endl;
        }

        // Decay learning rate
        if (lr_decay < 1.0f && (step + 1) % decay_every == 0) {
            current_lr *= lr_decay;
        }
    }
}

float Trainer::evaluate(Split split, int num_batches) {
    double total_loss = 0.0;
    for (int i = 0; i < num_batches; ++i) {
        Batch batch = dataset.get_batch(split);
        auto logits = model.forward(batch.inputs);
        total_loss += static_cast<double>(Loss::cross_entropy(logits, batch.targets));
    }
    return static_cast<float>(total_loss / static_cast<double>(num_batches));
}
