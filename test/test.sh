#!/bin/bash

# Run the main script
bash src/main.sh

# Check if results file exists
if [ -f "results/sim_results.csv" ]; then
    echo "Test passed: Results file found."
else
    echo "Test failed: No results file found."
    exit 1
fi
