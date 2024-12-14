#pragma once
#include <vector>
#include <string>

struct DataPoint {
    std::string country;
    int year;
    double population;
    double netMigration;
    double population_in_millions;
};

class Dataset {
public:
    Dataset(const std::string& filename);
    std::vector<DataPoint> load_data();
    void normalize_data(std::vector<DataPoint>& data);

private:
    std::string filename;
    int min_year, max_year;
    double min_migration, max_migration;
};
