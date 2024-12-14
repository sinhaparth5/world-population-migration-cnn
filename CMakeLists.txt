cmake_minimum_required(VERSION 3.10)
project(world_population_migration_cnn)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source File
set(SOURCES
    src/main.cpp
    src/data/dataset.cpp
    src/models/neural_network.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Add include directories
target_include_directories(${PROJECT_NAME} PRIVATE src)