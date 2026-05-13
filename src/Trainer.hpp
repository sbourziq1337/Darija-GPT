#pragma once

#include "Dataset.hpp"
#include "GPT.hpp"

// Orchestrates training loop for GPT model with learning rate decay
class Trainer {
private:
    Dataset& dataset;
    GPT& model;
    float learning_rate;
    float lr_decay;
    int decay_every;
    int print_every;

public:
    Trainer(Dataset& dataset, GPT& model, float learning_rate,
            float lr_decay = 1.0f, int decay_every = 1000, int print_every = 100);

    void train(int steps);
    float evaluate(Split split, int num_batches = 10);
};
