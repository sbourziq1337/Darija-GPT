#include "Trainer.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>

#include "Loss.hpp"

Trainer::Trainer(Dataset& dataset, GPT& model, float learning_rate,
                 float lr_decay, int decay_every, int print_every)
    : dataset(dataset),
      model(model),
      learning_rate(learning_rate),
      lr_decay(lr_decay),
      decay_every(decay_every),
      print_every(print_every),
      checkpoint_every_(0),
      best_loss_(9999.0f) {}

void Trainer::set_checkpoint_dir(const std::string& dir, int every_n_steps) {
    checkpoint_dir_ = dir;
    checkpoint_every_ = every_n_steps;
    if (!checkpoint_dir_.empty() && !std::filesystem::exists(checkpoint_dir_)) {
        std::filesystem::create_directories(checkpoint_dir_);
    }
}

void Trainer::save_checkpoint(int step, float lr, float loss) {
    if (checkpoint_dir_.empty()) return;

    CheckpointMetadata meta;
    meta.saved_step = step;
    meta.saved_lr = lr;
    meta.best_loss = best_loss_;

    std::string latest_path = checkpoint_dir_ + "/darija_gpt_latest.bin";
    std::string best_path = checkpoint_dir_ + "/darija_gpt_best.bin";

    // Always save latest
    if (!model.save(latest_path, meta)) {
        std::cerr << "Trainer: failed to save checkpoint to " << latest_path << std::endl;
        return;
    }

    // Save best if improved
    if (loss < best_loss_) {
        best_loss_ = loss;
        meta.best_loss = best_loss_;
        if (!model.save(best_path, meta)) {
            std::cerr << "Trainer: failed to save best checkpoint to " << best_path << std::endl;
        } else {
            std::cout << "  -> New best model saved (loss: " << best_loss_ << ")" << std::endl;
        }
    }
}

void Trainer::train(int steps, int start_step, float start_lr) {
    float current_lr = (start_lr > 0.0f) ? start_lr : learning_rate;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = start_step; step < steps; ++step) {
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
            float est_remaining = (steps - (step + 1)) * (elapsed / (step + 1 - start_step));

            std::cout << "Step " << (step + 1) << "/" << steps
                      << " | Loss: " << loss
                      << " | LR: " << current_lr
                      << " | " << step_time << "s/step"
                      << " | ~" << static_cast<int>(est_remaining / 60.0f) << "min left";

            if (best_loss_ < 9998.0f) {
                std::cout << " | Best: " << best_loss_;
            }
            std::cout << std::endl;
        }

        // Checkpoint every N steps
        if (checkpoint_every_ > 0 && (step + 1) % checkpoint_every_ == 0) {
            std::cout << "  -> Saving checkpoint at step " << (step + 1) << "..." << std::flush;
            save_checkpoint(step + 1, current_lr, loss);
            std::cout << " done." << std::endl;
        }

        // Update best loss even between checkpoints
        if (loss < best_loss_) {
            best_loss_ = loss;
        }

        // Decay learning rate
        if (lr_decay < 1.0f && (step + 1) % decay_every == 0) {
            current_lr *= lr_decay;
        }
    }

    // Final checkpoint
    if (checkpoint_every_ > 0 && steps > 0) {
        std::cout << "  -> Saving final checkpoint..." << std::flush;
        save_checkpoint(steps, current_lr, best_loss_);
        std::cout << " done." << std::endl;
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
