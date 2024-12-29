#!/bin/bash

# Exit on error
set -e

screen -S train_cl bash -c '
    # Activate conda environment
    eval "$(conda shell.bash hook)"
    conda activate cf

    # Run training scripts
    python ../train/train_permuted_mnist.py

    python ../train/train_split_cifar10.py

    python ../train/train_split_cifar100.py

    # Deactivate conda environment
    conda deactivate
'
