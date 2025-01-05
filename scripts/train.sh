#!/bin/bash

# Exit on error
set -e

#screen -S train_cl bash -c '
# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate cf

# Run training scripts
python ../train/train_permuted_mnist.py "training.num_subsamples_Hessian=2000"
python ../train/train_permuted_mnist.py "training.num_subsamples_Hessian=2000" "training.subspace_type='bulk'" "optimizer.calculate_next_top_k=False"
python ../train/train_permuted_mnist.py "training.num_subsamples_Hessian=2000" "training.subspace_type='dominant'" "optimizer.calculate_next_top_k=False" "optimizer.max_lanczos_steps=200"

python ../train/train_split_cifar10.py "data.batch_size=128" "optimizer.calculate_next_top_k=True"
python ../train/train_split_cifar10.py "training.subspace_type='bulk'" "data.batch_size=128" "optimizer.calculate_next_top_k=False"
python ../train/train_split_cifar10.py "training.subspace_type='dominant'" "data.batch_size=128" "optimizer.calculate_next_top_k=False" "optimizer.max_lanczos_steps=200"

python ../train/train_split_cifar100.py "data.batch_size=128" "optimizer.calculate_next_top_k=True"
python ../train/train_split_cifar100.py "training.subspace_type='bulk'" "data.batch_size=128" "optimizer.calculate_next_top_k=False"
python ../train/train_split_cifar100.py "training.subspace_type='dominant'" "data.batch_size=128" "optimizer.calculate_next_top_k=False" "optimizer.max_lanczos_steps=200"

# Deactivate conda environment
conda deactivate
#'
