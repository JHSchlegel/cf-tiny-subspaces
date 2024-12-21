#!/bin/bash

# Exit on error
set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate cf

# Run training script
python ../train/train_permuted_mnist.py

# Deactivate conda environment
conda deactivate
