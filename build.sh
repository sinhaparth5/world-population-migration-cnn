#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting build process...${NC}"

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo "Creating build directory..." 
    mkdir build
fi

# Navigate to build directory
cd build

echo "Running CMake..."
if cmake ..; then
    echo -e "${GREEN}CMake configuration successful${NC}"
else
    echo -e "${RED}CMake configuration failed${NC}"
    exit 1
fi

# Run make
echo "Building project..."
if make; then
    echo -e "${GREEN}Build successful${NC}"
else
    echo -e "${RED}Build failed${NC}"
    exit 1
fi

# Run the executable
echo -e "${GREEN}Running the program...${NC}"
./world_population_migration_cnn

# Return to original directory
cd ..