#include "neural_network.hpp"
#include <random>
#include <algorithm>
#include <cmath>
#include <stdexcept>

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, double learning_rate) 
    : input_size(input_size), hidden_size(hidden_size), learning_rate(learning_rate) {
    
    // Initialize with smaller weights using He initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    double scale = std::sqrt(2.0 / input_size);
    std::normal_distribution<> d(0, scale);
    
    // Initialize weights with smaller values
    input_weights.resize(input_size * hidden_size);
    input_biases.resize(hidden_size);
    for (int i = 0; i < input_size * hidden_size; ++i) {
        input_weights[i] = d(gen) * 0.1;  // Scale down initial weights
    }
    for (int i = 0; i < hidden_size; ++i) {
        input_biases[i] = 0.0;  // Start with zero bias
    }
    
    hidden_weights.resize(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        hidden_weights[i] = d(gen) * 0.1;  // Scale down initial weights
    }
    output_bias = 0.0;
    
    // Initialize cache vectors
    hidden_values.resize(hidden_size);
    hidden_raw.resize(hidden_size);
}

double NeuralNetwork::activate(double x) {
    // Use tanh activation for better numerical stability
    return std::tanh(x);
}

double NeuralNetwork::activate_derivative(double x) {
    // Derivative of tanh
    double tanh_x = std::tanh(x);
    return 1.0 - tanh_x * tanh_x;
}

double NeuralNetwork::clip(double x, double min_val, double max_val) {
    return std::min(std::max(x, min_val), max_val);
}

double NeuralNetwork::forward(const std::vector<double>& inputs) {
    if (inputs.size() != input_size) {
        throw std::runtime_error("Invalid input size");
    }
    
    // Forward pass through hidden layer
    for (int i = 0; i < hidden_size; ++i) {
        hidden_raw[i] = input_biases[i];
        for (int j = 0; j < input_size; ++j) {
            hidden_raw[i] += inputs[j] * input_weights[i * input_size + j];
        }
        hidden_raw[i] = clip(hidden_raw[i], -100.0, 100.0);  // Prevent overflow
        hidden_values[i] = activate(hidden_raw[i]);
    }
    
    // Output layer
    double output = output_bias;
    for (int i = 0; i < hidden_size; ++i) {
        output += hidden_values[i] * hidden_weights[i];
    }
    
    return clip(output, 0.0, 1.0);  // Ensure output is in valid range
}

void NeuralNetwork::train(const std::vector<double>& inputs, double target) {
    // Forward pass
    double output = forward(inputs);
    
    // Clip gradients for numerical stability
    double error = clip(target - output, -1.0, 1.0);
    double output_delta = error * 0.1;  // Scale down the error
    
    // Update hidden layer weights with gradient clipping
    for (int i = 0; i < hidden_size; ++i) {
        double gradient = learning_rate * output_delta * hidden_values[i];
        gradient = clip(gradient, -0.1, 0.1);  // Clip gradients
        hidden_weights[i] += gradient;
    }
    output_bias += learning_rate * output_delta;
    
    // Hidden layer gradients
    std::vector<double> hidden_deltas(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        hidden_deltas[i] = output_delta * hidden_weights[i] * 
            activate_derivative(hidden_raw[i]);
        hidden_deltas[i] = clip(hidden_deltas[i], -1.0, 1.0);  // Clip gradients
    }
    
    // Update input weights with momentum
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            double gradient = learning_rate * hidden_deltas[i] * inputs[j];
            gradient = clip(gradient, -0.1, 0.1);  // Clip gradients
            input_weights[i * input_size + j] += gradient;
        }
        input_biases[i] += learning_rate * hidden_deltas[i];
    }
}