#pragma once

#include <string>
#include "Checkpoint.hpp"
#include "Dataset.hpp"
#include "GPT.hpp"

// Orchestrates training loop for GPT model with learning rate decay and checkpointing
class Trainer {
private:
    Dataset& dataset;
    GPT& model;
    float learning_rate;
    float lr_decay;
    int decay_every;
    int print_every;

    // Checkpoint settings
    std::string checkpoint_dir_;
    int checkpoint_every_;
    float best_loss_;

public:
    Trainer(Dataset& dataset, GPT& model, float learning_rate,
            float lr_decay = 1.0f, int decay_every = 1000, int print_every = 100);

    // Set checkpoint directory and frequency (every N steps)
    void set_checkpoint_dir(const std::string& dir, int every_n_steps = 500);

    // Train for given steps, resuming from a specific step if needed
    void train(int steps, int start_step = 0, float start_lr = 0.0f);

    float evaluate(Split split, int num_batches = 10);

    // Get best loss seen so far
    float best_loss() const { return best_loss_; }

private:
    void save_checkpoint(int step, float lr, float loss);
};
