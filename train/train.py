import torch
import torch.nn as nn
import argparse
import yaml
from pathlib import Path
import logging
from torch.utils.data import DataLoader


from modules.CLTrainer import CLTrainer
from utils.wandb_utils import setup_wandb
from modules.mlp import MLP
from modules.cnn import CNN
from modules.subspace_sgd import SubspaceSGD



def train_and_evaluate(config_path: str):
    """
    Main training and evaluation function.
    
    Args:
        config_path: Path to the config.yaml file of MLP/CNN
    """
    #TODO config.yaml?
    with open(config_path) as f:
        config = yaml.safe_load(f)

    #TODO standard logger. maybe change
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    #TODO Dataloader
    train_loader, test_loader = get_data_loaders(config)

    #TODO Do we get the model from config.yaml?
    model = get_model(config)
    optimizer = SubspaceSGD(
        model.parameters(),
        **config['optimizer']['params']
    )
    criterion = nn.CrossEntropyLoss()

    # Call the trainer
    trainer = CLTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        epoch=config['training']['n_epochs'],
        log_interval=config['training']['log_interval'],
        subspace_type=config['training']['subspace_type'],
        setup_wandb=config['wandb']['enabled'],
        device=config['training'].get('device', 'cuda')
    )

    # Training loop. Note that for the first epoch standard SGD is applied 
    best_accuracy = 0.0
    for epoch in range(config['training']['n_epochs']):
        # Train one step
        train_loss, train_acc, train_time, eigenvalues = trainer.step(
            train_loader=train_loader,
            epoch=epoch
        )

        logging.info(f"Epoch {epoch+1}/{config['training']['n_epochs']}")
        logging.info(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # Evaluation
        if (epoch + 1) % config['training']['eval_freq'] == 0:
            eval_acc, eval_loss = trainer.evaluate(
                test_loader=test_loader,
                prefix='eval'
            )
            
            logging.info(f"Evaluation - Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.2f}%")

            # Save best model
            if eval_acc > best_accuracy:
                best_accuracy = eval_acc
                save_path = Path(config['training']['checkpoint_dir']) / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_accuracy,
                    'config': config
                }, save_path)
                logging.info(f"Saved best model with accuracy: {best_accuracy:.2f}%")


def main():
    try:
        #TODO get the config.
        train_and_evaluate(config)
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()


#TODO
# 1. Additional metrics; e.g., to quantify forgetting
# 2. Fix Wandb or some more logging
# 3. Eigenvalues can be used efficiently. here we just call them