#include "dataset.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

Dataset::Dataset(const std::string& filename) : filename(filename), min_year(0), max_year(0), min_migration(0), max_migration(0) {}

std::vector<DataPoint> Dataset::load_data() {
    std::vector<DataPoint> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        DataPoint dp;

        std::getline(ss, dp.country, ',');
        if (!std::getline(ss, field, ',') || field.empty()) continue;
        dp.year = std::stoi(field);

        if (!std::getline(ss, field, ',') || field.empty()) continue;
        dp.population = std::stod(field);

        if (!std::getline(ss, field, ',') || field.empty()) continue;
        dp.netMigration = std::stod(field);

        if (!std::getline(ss, field, ',') || field.empty()) continue;
        dp.population_in_millions = std::stod(field);

        data.push_back(dp);
    }

    return data;
}

void Dataset::normalize_data(std::vector<DataPoint>& data) {
    if (data.empty()) return;

    min_year = max_year = data[0].year;
    min_migration = max_migration = data[0].netMigration;

    for (const auto& dp : data) {
        min_year = std::min(min_year, dp.year);
        max_year = std::max(max_year, dp.year);
        min_migration = std::min(min_migration, dp.netMigration);
        max_migration = std::max(max_migration, dp.netMigration);
    }

    for (auto& dp : data) {
        dp.year = static_cast<int>((dp.year - min_year) / static_cast<double>(max_year - min_year));
        dp.netMigration = (dp.netMigration - min_migration) / (max_migration - min_migration);
    }
}
