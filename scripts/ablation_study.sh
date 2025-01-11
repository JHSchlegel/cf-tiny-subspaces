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

# Dataset to run ablation on: pmnist, cifar10, cifar100
if [ -z "$1" ]; then
    echo "Please provide a dataset name (pmnist, cifar10, or cifar100)"
    exit 1
fi
DATASET="$1"

# Hyperparameter ranges:
case $DATASET in
"pmnist")
    # Permuted MNIST hyperparameters
    WIDTHS=(50 200)            # default: 100; was already run separately
    BATCH_SIZES=(256 512)      # default: 128; was already run separately
    LEARNING_RATES=(0.001 0.1) # default: 0.01; was already run separately
    DIM_PARAM_NAME="hidden_dim"
    DEFAULT_WIDTH=100
    DEFAULT_BS=128
    DEFAULT_LR=0.01
    ;;
"cifar10" | "cifar100")
    # CIFAR hyperparameters
    WIDTHS=(16 64)               # default: 32; was already run separately
    BATCH_SIZES=(256 512)        # default: 128; was already run separately
    LEARNING_RATES=(0.0001 0.01) # default: 0.001; was already run separately
    DIM_PARAM_NAME="width"
    DEFAULT_WIDTH=32
    DEFAULT_BS=128
    DEFAULT_LR=0.001
    ;;
*)
    echo "Invalid dataset name. Please use pmnist, cifar10, or cifar100"
    exit 1
    ;;
esac

# Function to run a single dataset experiment
run_dataset_experiment() {
    local dataset=$1
    local width=$2
    local batch_size=$3
    local lr=$4
    local exp_name="width-${width}_bs-${batch_size}_lr-${lr}"
    local num_epochs=$((5 * batch_size / 128))

    case $dataset in
    "pmnist")
        script_path="../train/train_permuted_mnist.py"
        dim_param="model.hidden_dim=${width}"
        task_params="data.num_tasks=10"
        ;;
    "cifar10")
        script_path="../train/train_split_cifar10.py"
        dim_param="model.width=${width}"
        task_params="data.num_tasks=5 data.classes_per_task=2"
        ;;
    "cifar100")
        script_path="../train/train_split_cifar100.py"
        dim_param="model.width=${width}"
        task_params="data.num_tasks=10 data.classes_per_task=10"
        ;;
    esac

    log_message "Running ${dataset} with configuration: ${exp_name}"
    log_message "Number of epochs: $num_epochs"
    log_message "Script path: $script_path"

    # Hydra overrides:
    python ${script_path} \
        ${dim_param} \
        data.batch_size=${batch_size} \
        optimizer.lr=${lr} \
        optimizer.calculate_next_top_k=False \
        training.subspace_type='bulk' \
        training.calculate_overlap=True \
        training.num_subsamples_Hessian=2000 \
        training.num_epochs=${num_epochs} \
        wandb.project="ablation_${dataset}_${exp_name}" \
        hydra.run.dir="$BASE_DIR/${dataset}/${exp_name}" \
        hydra.sweep.dir="$BASE_DIR/${dataset}/${exp_name}" \
        hydra/job_logging=disabled \
        hydra.job.chdir=True
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
    shift 2
    local param_values=("$@")

    log_message "Starting ablation study for $param_name"

    for value in "${param_values[@]}"; do
        case $param_name in
        "width")
            run_configuration "$value" "$DEFAULT_BS" "$DEFAULT_LR" "$dataset"
            ;;
        "batch_size")
            run_configuration "$DEFAULT_WIDTH" "$value" "$DEFAULT_LR" "$dataset"
            ;;
        "learning_rate")
            run_configuration "$DEFAULT_WIDTH" "$DEFAULT_BS" "$value" "$dataset"
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
