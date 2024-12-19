import os
import torch
import torch.nn as nn
import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from modules.CLTrainer import CLTrainer
from modules.mlp import MLP
from modules.cnn import CNN
from modules.subspace_sgd import SubspaceSGD
from utils.data_utils import get_data_loaders
from utils.model_utils import get_model

# Ignore annoying hydra warnings
warnings.filterwarnings("ignore")

@hydra.main(
    config_path="../configs/model", 
    config_name="trainer",  
    version_base=None
)
def main(config: DictConfig) -> None:
    """
    Main training function with Hydra configuration.
    
    Args:
        config: Hydra configuration object containing all parameters
    """
    # Save directory is automatically managed by Hydra
    save_dir = Path(os.getcwd())
    print(f"Saving results to {save_dir}")

    # Get data loaders
    train_loader, test_loader = get_data_loaders(config.data)

    # Initialize model
    model = get_model(config.model)
    
    # Initialize optimizer
    optimizer = SubspaceSGD(
        model.parameters(),
        **OmegaConf.to_container(config.optimizer, resolve=True)
    )
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize trainer
    trainer = CLTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        save_dir=str(save_dir),
        epoch=config.training.n_epochs,
        log_interval=config.training.log_interval,
        subspace_type=config.training.subspace_type,
        setup_wandb=config.wandb.enabled,
        wandb_project=config.wandb.get('project', None),
        wandb_config=OmegaConf.to_container(config, resolve=True),
        device=config.training.get('device', 'cuda'),
        eval_freq=config.training.get('eval_freq', 1)
    )

    try:
        # Train and evaluate
        train_losses, train_accuracies, final_eval_loss, final_eval_acc, eigenvalues = trainer.train_and_evaluate(
            train_loader=train_loader,
            test_loader=test_loader
        )

        print(f"Training completed. Final test accuracy: {final_eval_acc:.2f}%")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise
    finally:
        # Clean up GPU memory
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()