"""
Training script for Permuted MNIST experiments.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import os
import sys
import torch
import torch.nn as nn
import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.abspath("../"))
from modules.CLTrainer import CLTrainer
from modules.mlp import MLP
from modules.subspace_sgd import SubspaceSGD
from utils.data_utils.permuted_mnist import PermutedMNIST

# Ignore annoying hydra warnings
warnings.filterwarnings("ignore")


# =========================================================================== #
#                          Main Training Function                             #
# =========================================================================== #
@hydra.main(config_path="../configs", config_name="permuted_mnist")
def main(config: DictConfig) -> None:
    """
    Main training function with Hydra configuration.

    Args:
        config: Hydra configuration object containing all parameters
    """
    # Save directory is automatically managed by Hydra
    save_dir = f"{os.getcwd()}/"
    print(f"Saving results to {save_dir}")

    # Create permuted MNIST dataset:
    pmnist = PermutedMNIST(num_tasks=config.data.num_tasks, seed=config.data.seed)
    pmnist.setup_tasks(
        batch_size=config.data.batch_size,
        data_root=config.data.get("data_root", "./data"),
        num_workers=config.data.get("num_workers", 4),
    )

    # Initialize model
    model = MLP(
        input_dim=config.model.input_dim,
        output_dim=config.model.output_dim,
        hidden_dim=config.model.hidden_dim,
    )
    model.to(config.training.get("device", "cuda"))

    # Initialize loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer
    optimizer = SubspaceSGD(
        model,
        criterion=criterion,
        **OmegaConf.to_container(config.optimizer, resolve=True),
    )

    # Initialize trainer
    trainer = CLTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        save_dir=str(save_dir),
        num_tasks=config.data.num_tasks,
        num_epochs=config.training.num_epochs,
        log_interval=config.training.log_interval,
        eval_freq=config.training.get("eval_freq", 1),
        task_il=False,
        calculate_overlap=config.training.get("calculate_overlap", True),
        num_subsamples_Hessian=config.training.get("num_subsamples_Hessian", 5_000),
        checkpoint_freq=config.training.get("checkpoint_freq", 10),
        seed=config.training.seed,
        subspace_type=config.training.subspace_type,
        scheduler=None,
        device=config.training.get("device", "cuda"),
        use_wandb=config.wandb.enabled,
        wandb_project=config.wandb.get("project", None),
        wandb_config=OmegaConf.to_container(config, resolve=True),
    )

    try:
        # Train and evaluate
        (
            avg_accuracy,
            avg_max_forgetting,
            train_losses,
            train_accuracies,
            test_accuracies,
            test_losses,
            eigenvalues,
        ) = trainer.train_and_evaluate(pmnist)

        print(f"Test accuracies: {test_accuracies}")
        print(f"Test losses: {test_losses}")

    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise
    finally:
        # Clean up GPU memory
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
