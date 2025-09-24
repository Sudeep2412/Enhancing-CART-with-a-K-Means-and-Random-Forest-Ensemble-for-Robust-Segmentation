#!/bin/bash

# Source the Conda initialization script
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the target environment
conda activate rapids-25.08

# Run the Python script using the full path to the interpreter
/home/skiive/miniconda3/envs/rapids-25.08/bin/python model1.py