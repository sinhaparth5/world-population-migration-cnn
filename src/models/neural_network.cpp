#include "neural_network.hpp"
#include <random>
#include <algorithm>
#include <cmath>
#include <stdexcept>

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, double learning_rate) 
    : input_size(input_size), hidden_size(hidden_size), learning_rate(learning_rate),
      input_weights(input_size * hidden_size), hidden_weights(hidden_size),
      input_biases(hidden_size), hidden_values(hidden_size), hidden_raw(hidden_size) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, std::sqrt(2.0 / input_size));

    for (auto& weight : input_weights) weight = d(gen);
    for (auto& weight : hidden_weights) weight = d(gen);
    std::fill(input_biases.begin(), input_biases.end(), 0.0);
    output_bias = 0.0;
}

double NeuralNetwork::activate(double x) {
    return std::tanh(x);
}

double NeuralNetwork::activate_derivative(double x) {
    double tanh_x = std::tanh(x);
    return 1.0 - tanh_x * tanh_x;
}

double NeuralNetwork::forward(const std::vector<double>& inputs) {
    if (inputs.size() != input_size) {
        throw std::runtime_error("Invalid input size");
    }

    for (int i = 0; i < hidden_size; ++i) {
        hidden_raw[i] = input_biases[i];
        for (int j = 0; j < input_size; ++j) {
            hidden_raw[i] += inputs[j] * input_weights[i * input_size + j];
        }
        hidden_values[i] = activate(hidden_raw[i]);
    }

    double output = output_bias;
    for (int i = 0; i < hidden_size; ++i) {
        output += hidden_values[i] * hidden_weights[i];
    }

    return output;
}

void NeuralNetwork::train(const std::vector<double>& inputs, double target) {
    double output = forward(inputs);
    double error = target - output;

    // Output layer gradients
    double output_delta = error;
    for (int i = 0; i < hidden_size; ++i) {
        hidden_weights[i] += learning_rate * output_delta * hidden_values[i];
    }
    output_bias += learning_rate * output_delta;

    // Hidden layer gradients
    for (int i = 0; i < hidden_size; ++i) {
        double hidden_delta = output_delta * hidden_weights[i] * activate_derivative(hidden_raw[i]);
        for (int j = 0; j < input_size; ++j) {
            input_weights[i * input_size + j] += learning_rate * hidden_delta * inputs[j];
        }
        input_biases[i] += learning_rate * hidden_delta;
    }
}
