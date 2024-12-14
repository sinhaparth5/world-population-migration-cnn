#pragma once
#include <vector>
#include <stdexcept>

class NeuralNetwork {
public:
    NeuralNetwork(int input_size = 2, int hidden_size = 16, double learning_rate = 0.001);
    double forward(const std::vector<double>& inputs);
    void train(const std::vector<double>& inputs, double target);

private:
    std::vector<double> input_weights;
    std::vector<double> hidden_weights;
    std::vector<double> input_biases;
    double output_bias;

    // Cache for backpropagation
    std::vector<double> hidden_values;
    std::vector<double> hidden_raw;

    double learning_rate;
    int input_size;
    int hidden_size;

    double activate(double x);
    double activate_derivative(double x);
};
