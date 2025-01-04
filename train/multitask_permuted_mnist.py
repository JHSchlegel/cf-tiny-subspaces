# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import os
import sys
import torch
import torch.nn as nn
import warnings
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.abspath("../"))
from modules.MTLTrainer import MTLTrainer
from modules.mlp import MLP
from modules.subspace_sgd import SubspaceSGD
from utils.data_utils.permuted_mnist import PermutedMNIST

# Ignore annoying hydra warnings
warnings.filterwarnings("ignore")


# =========================================================================== #
#                          Main Training Function                             #
# =========================================================================== #
@hydra.main(config_path="../configs", config_name="multitask_permuted_mnist")
def main(config: DictConfig) -> None:
    """
    Main training function with Hydra configuration for unified multi-task learning.
    """
    save_dir = f"{os.getcwd()}/"
    print(f"Saving results to {save_dir}")

    # Create permuted MNIST datasets but combine them into one loader
    pmnist = PermutedMNIST(num_tasks=config.data.num_tasks, seed=config.data.seed)
    pmnist.setup_tasks(
        batch_size=config.data.batch_size,
        data_root=config.data.get("data_root", "./data"),
        num_workers=config.data.get("num_workers", 4),
    )
    
    # Combine all task datasets into one
    combined_train_dataset = ConcatDataset([
        pmnist.train_loaders[task_id].dataset for task_id in range(config.data.num_tasks)
    ])
    combined_test_dataset = ConcatDataset([
        pmnist.test_loaders[task_id].dataset for task_id in range(config.data.num_tasks)
    ])

    # Create unified dataloaders
    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        combined_test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
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
        model=model,
        criterion=criterion,
        **OmegaConf.to_container(optimizer.config, resolve=True),
    )

    # Initialize trainer
    trainer = MTLTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        save_dir=str(save_dir),
        num_epochs=config.training.num_epochs,
        log_interval=config.training.log_interval,
        eval_freq=config.training.get("eval_freq", 1),
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
        train_metrics, val_metrics = trainer.train_and_evaluate(
            train_loader=train_loader,
            val_loader=val_loader,
        )

        print("Training completed successfully!")
        print(f"Final training accuracy: {train_metrics['accuracies'][-1]:.2f}%")
        print(f"Final validation accuracy: {val_metrics['accuracies'][-1]:.2f}%")

    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()