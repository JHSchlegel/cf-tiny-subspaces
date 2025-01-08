#!/bin/bash

# Exit on error
set -e

# Base output directory
BASE_DIR="../outputs"
mkdir -p "$BASE_DIR"

# Log file setup
LOG_FILE="ablation_study.log"
touch "$LOG_FILE"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Hyperparameter ranges
## Permuted MNIST:
# HIDDEN_DIMS=(50 200) # default: 100; was already run separately
# BATCH_SIZES=(256 512) # default: 128; was already run separately
# LEARNING_RATES=(0.001 0.1) # default: 0.01; was already run separately

## Split CIFAR:
WIDTHS=(16 64)               # default: 32; was already run separately
BATCH_SIZES=(256 512)        # default: 128; was already run separately
LEARNING_RATES=(0.0001 0.01) # default: 0.001; was already run separately

# Dataset to run ablation on: --pmnist, --cifar10, --cifar100
DATASET="$1"

# Function to run a single dataset experiment
run_dataset_experiment() {
    local dataset=$1
    local hidden_dim=$2
    local batch_size=$3
    local lr=$4
    local exp_name="hd-${hidden_dim}_bs-${batch_size}_lr-${lr}"
    local num_epochs=$((5 * batch_size / 128))

    case $dataset in
    "pmnist")
        script_path="../train/train_permuted_mnist.py"
        config_name="permuted_mnist"
        ;;
    "cifar10")
        script_path="../train/train_split_cifar10.py"
        config_name="split_cifar10"
        ;;
    "cifar100")
        script_path="../train/train_split_cifar100.py"
        config_name="split_cifar100"
        ;;
    esac

    log_message "Running ${dataset} with configuration: ${exp_name}"
    log_message "Number of epochs: $num_epochs"
    log_message "Script path: $script_path"

    # Using Hydra's multirun feature with structured overrides
    # model.hidden_dim=${hidden_dim} \
    python ${script_path} \
        model.width=32 \
        data.batch_size=${batch_size} \
        optimizer.lr=${lr} \
        optimizer.calculate_next_top_k=False \
        optimizer.k=10 \
        training.subspace_type=null \
        training.calculate_overlap=True \
        training.num_subsamples_Hessian=2000 \
        data.num_tasks=10 \
        training.num_epochs=${num_epochs} \
        wandb.project="ablation_${dataset}_${exp_name}" \
        hydra.run.dir="$BASE_DIR/${dataset}/${exp_name}" \
        hydra.sweep.dir="$BASE_DIR/${dataset}/${exp_name}" \
        hydra/job_logging=disabled \
        hydra.job.chdir=True
}

# Function to run experiments for all datasets with one configuration
run_configuration() {
    local hidden_dim=$1
    local batch_size=$2
    local lr=$3
    local dataset=$4
    local exp_name="hd-${hidden_dim}_bs-${batch_size}_lr-${lr}"

    log_message "Starting experiments for configuration: ${exp_name}"

    #screen -S "cl_ablation_${exp_name}" -d -m bash -c "
    # Activate conda environment
    eval "$(conda shell.bash hook)"
    conda activate cf

    run_dataset_experiment $dataset $hidden_dim $batch_size $lr

    conda deactivate
}

# Function to run ablation for a single hyperparameter
run_ablation() {
    local param_name=$1
    local dataset=$2
    shift 2
    local param_values=("$@")

    log_message "Starting ablation study for $param_name"

    # Default values
    local default_hidden_dim=100
    local default_batch_size=128
    local default_lr=0.01

    for value in "${param_values[@]}"; do
        case $param_name in
        "hidden_dim")
            run_configuration "$value" "$default_batch_size" "$default_lr" "$dataset"
            ;;
        "batch_size")
            run_configuration "$default_hidden_dim" "$value" "$default_lr" "$dataset"
            ;;
        "learning_rate")
            run_configuration "$default_hidden_dim" "$default_batch_size" "$value" "$dataset"
            ;;
        esac
    done
}

# Function to run experiments for all datasets with one configuration
run_configuration() {
    local width=$1
    local batch_size=$2
    local lr=$3
    local dataset=$4
    local exp_name="width-${width}_bs-${batch_size}_lr-${lr}"

    log_message "Starting experiments for configuration: ${exp_name}"

    #screen -S "cl_ablation_${exp_name}" -d -m bash -c "
    # Activate conda environment
    eval "$(conda shell.bash hook)"
    conda activate cf

    run_dataset_experiment $dataset $width $batch_size $lr

    conda deactivate
}

# Function to run ablation for a single hyperparameter
run_ablation() {
    local param_name=$1
    local dataset=$2
    local param_values=("${@:3}")

    log_message "Starting ablation study for $param_name"

    # Default values
    local default_width=32
    local default_batch_size=128
    local default_lr=0.001

    for value in "${param_values[@]}"; do
        case $param_name in
        "width")
            run_configuration "$value" "$default_batch_size" "$default_lr" "$dataset"
            ;;
        "batch_size")
            run_configuration "$default_width" "$value" "$default_lr" "$dataset"
            ;;
        "learning_rate")
            run_configuration "$default_width" "$default_batch_size" "$value" "$dataset"
            ;;
        esac
    done
}

# Main execution
log_message "Starting comprehensive ablation study"

# Run individual ablations
log_message "Running width dimension ablation..."
run_ablation "width" "$DATASET" "${WIDTHS[@]}"

log_message "Running batch size ablation..."
run_ablation "batch_size" "$DATASET" "${BATCH_SIZES[@]}"

log_message "Running learning rate ablation..."
run_ablation "learning_rate" "$DATASET" "${LEARNING_RATES[@]}"

log_message "Ablation study completed. Thanks for your patience :)"
