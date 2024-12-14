#include "data/dataset.hpp"
#include "models/neural_network.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>

int main() {
    try {
        // Load and prepare data
        Dataset dataset("../data/world_pop_mig_186_countries.csv");
        auto data = dataset.load_data();
        dataset.normalize_data(data);
        
        // Create neural network with smaller learning rate
        NeuralNetwork nn(2, 16, 0.001);
        
        // Training
        const int epochs = 1000;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_error = 0.0;
            
            for (const auto& dp : data) {
                // Scale inputs to [0,1] range
                std::vector<double> inputs = {
                    static_cast<double>(dp.year),
                    (dp.netMigration + 1000000) / 2000000  // Assume migration in [-1M, 1M] range
                };
                
                // Scale target to [0,1] range
                double target = dp.population_in_millions / 100.0;  // Assume max population 100M
                
                nn.train(inputs, target);
                
                double prediction = nn.forward(inputs);
                total_error += std::pow(prediction - target, 2);
            }
            
            if (epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Error: " 
                         << std::fixed << std::setprecision(6) 
                         << total_error / data.size() << std::endl;
            }
        }
        
        // Test predictions
        std::cout << "\nTesting the network:\n";
        for (int i = 0; i < 5; ++i) {
            std::vector<double> inputs = {
                static_cast<double>(data[i].year),
                (data[i].netMigration + 1000000) / 2000000
            };
            
            double prediction = nn.forward(inputs) * 100.0;  // Scale back to millions
            std::cout << "Country: " << data[i].country 
                     << ", Year: " << data[i].year 
                     << ", Actual: " << data[i].population_in_millions 
                     << ", Predicted: " << std::fixed << std::setprecision(2) 
                     << prediction << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}