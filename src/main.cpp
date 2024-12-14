#include "data/dataset.hpp"
#include "models/neural_network.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>

int main() {
    try {
        // Load and prepare the dataset
        Dataset dataset("../data/world_pop_mig_186_countries.csv");
        auto data = dataset.load_data();
        dataset.normalize_data(data);

        // Define neural network parameters
        const int input_size = 2;
        const int hidden_size = 16;
        const double learning_rate = 0.001;

        NeuralNetwork nn(input_size, hidden_size, learning_rate);

        // Training loop
        const int epochs = 1000;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_error = 0.0;

            for (const auto& dp : data) {
                // Normalize input data to the [0, 1] range
                std::vector<double> inputs = {
                    static_cast<double>(dp.year),
                    (dp.netMigration + 1000000) / 2000000  // Migration normalized to [-1M, 1M]
                };

                // Normalize target value to the [0, 1] range
                double target = dp.population_in_millions / 100.0;  // Assume max population 100M

                // Train the network
                nn.train(inputs, target);

                // Calculate the squared error for logging
                double prediction = nn.forward(inputs);
                total_error += std::pow(prediction - target, 2);
            }

            // Log progress every 100 epochs
            if (epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Mean Squared Error: "
                          << std::fixed << std::setprecision(6)
                          << total_error / data.size() << std::endl;
            }
        }

        // Testing the trained network on a subset of data
        std::cout << "\nTesting the network:\n";
        std::string target_country = "India"; // Change to the desired country

        size_t count = 0;
        for (const auto& dp : data) {
            if (dp.country != target_country && !target_country.empty()) {
                continue; // Skip data not matching the target country
            }

            std::vector<double> inputs = {
                static_cast<double>(dp.year),
                (dp.netMigration + 1000000) / 2000000  // Normalize migration
            };

            // Generate prediction and scale it back to the original range
            double prediction = nn.forward(inputs) * 100.0;  // Denormalize prediction

            std::cout << "Country: " << dp.country
                      << ", Year: " << dp.year
                      << ", Actual Population (in millions): " << dp.population_in_millions
                      << ", Predicted Population (in millions): " << std::fixed << std::setprecision(2)
                      << prediction << std::endl;

            if (++count >= 5) break; // Display up to 5 predictions
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
