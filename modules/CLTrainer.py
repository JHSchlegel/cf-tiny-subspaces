import time
from tqdm import tqdm
from typing import Tuple, Optional, Dict, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging

import os
import sys
import wandb

from utils.wandb_utils import setup_wandb



class CLTrainer:
    """
    Subspace-restricted Trainer implementation in Continual Learning setting that supports 
    training with externally configured optimizers. This class provides comprehensive
    training, evaluation, and metric tracking capabilities integrated with wandb logging.
    """

    VALID_SUBSPACE_TYPES = {'bulk', 'dominant', None}   

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        save_dir: str,
        epoch: int,
        log_interval: int,
        eval_freq: int,
        seed: int = 42,
        subspace_type: Optional[str] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        setup_wandb: bool = True,   
        wandb_log_dir: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
    ) -> None:
        """
        Initialize an instance of the Continual Learning Trainer.

        Args:
            model: Neural network model being trained
            optimizer: Custom-built subspace optimizer
            criterion: Loss function used for training
            save_dir: Directory path to save models and visualizations
            epoch: Number of epochs for training
            log_interval: Interval for logging metrics during training
            eval_freq: Frequency of evaluation on the test set
            seed: Random seed for reproducibility
            subspace_type: Type of subspace projection ("bulk" or "dominant")
            scheduler: Learning rate scheduler
            device: Device to run computations on
            setup_wandb: Whether to initialize wandb logging
            wandb_log_dir: Logging directory for weights and biases
            wandb_project: Project name of weights and biases
            wandb_config: Configs used for run; saved in weights and biases
        """
        if subspace_type not in self.VALID_SUBSPACE_TYPES:
            raise ValueError(f"subspace_type must be one of {self.VALID_SUBSPACE_TYPES}")

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.epoch = epoch
        self.log_interval = log_interval
        self.eval_freq = eval_freq
        self.subspace_type = subspace_type
        self.device = device

        # Create directories to save checkpoints
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "models").mkdir(exist_ok=True)

        # Set seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
                
        self.setup_wandb: bool = setup_wandb
        if setup_wandb:
            setup_wandb(
                dir=wandb_log_dir,
                project=wandb_project,
                config=wandb_config
            )

        logging.basicConfig(level=logging.INFO)
        
    def step(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Tuple[float, float, float, List[torch.Tensor]]:
        """
        Train the model for one epoch with or without subspace projection.

        Args:
            train_loader (DataLoader): DataLoader containing the current task's training data
            epoch (int): Current epoch number for training

        Returns:
            Tuple containing:
                float: Average loss value for the epoch
                float: Accuracy percentage for the epoch
                float: Total training time for the epoch in seconds
                List[torch.Tensor]: List of eigenvalues computed during training
        """
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        eigenvalues_list = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()

            # Use the custom built optimizer. We pass a standard optimization
            # in the first epoch and perform gradient projection in later steps.
            self.optimizer.step(
                data_batch=(data, target),
                fp16=False,
                subspace_type=None if epoch==0 else self.subspace_type
            )

            # Track eigenvalues
            if self.subspace_type:
                eigenvalues, _ = self.optimizer.eigenthings
                eigenvalues_list.append(eigenvalues)

            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += torch.sum(pred.eq(target.view_as(pred))).item()
            total += target.size(0)
            epoch_loss += loss.item()

            # Log progress in wandb
            if batch_idx % self.log_interval == 0:
                current_progress = batch_idx * len(data)
                total_samples = len(train_loader.dataset)
                progress_percentage = 100. * batch_idx / len(train_loader)
                
                logging.info(
                    f"Train Epoch: {epoch} [{current_progress}/{total_samples} "
                    f"({progress_percentage:.0f}%)]\t"
                    f"Loss: {loss.item():.6f}"
                )

        # Calculate epoch statistics
        epoch_time = time.time() - start_time
        epoch_loss /= len(train_loader)
        epoch_accuracy = 100.0 * correct / total

        # Log epoch metrics
        if wandb.run:
            wandb.log({
                'train/epoch_loss': epoch_loss,
                'train/epoch_accuracy': epoch_accuracy,
                'train/epoch_time': epoch_time,
                'train/epoch': epoch
            })

            # Log eigenvalue statistics
            if eigenvalues_list:
                last_eigenvalues = eigenvalues_list[-1]
                wandb.log({
                    'train/top_eigenvalue': last_eigenvalues[0],
                    'train/eigenvalue_ratio': last_eigenvalues[0] / last_eigenvalues[-1],
                    'train/epoch': epoch
                })

        return epoch_loss, epoch_accuracy, epoch_time, eigenvalues_list    

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        prefix: str = 'eval',
    ) -> Tuple[float, float]:
        """
        Evaluate the model's performance on a test dataset and log metrics to wandb.
        
        Args:
            test_loader (DataLoader): DataLoader containing the test/validation data
            prefix (str, optional): Prefix for wandb logging keys. Defaults to 'eval'.
        
        Returns:
            Tuple containing:
                float: Model accuracy as a percentage
                float: Average loss value across all test samples
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        # Track time for evaluation
        eval_start_time = time.time()

        pbar = tqdm(test_loader, desc='Evaluating...')
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Accumulate loss
            total_loss += loss.item() * data.size(0)
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += torch.sum(pred.eq(target.view_as(pred))).item()
            total_samples += data.size(0)
            
            # Update progress bar
            current_acc = 100. * correct / total_samples
            current_loss = total_loss / total_samples
            pbar.set_postfix({
                f'{prefix}/loss': f'{current_loss:.4f}',
                f'{prefix}/acc': f'{current_acc:.2f}%'
            })

        # Calculate final metrics
        eval_time = time.time() - eval_start_time
        avg_loss = total_loss / total_samples
        accuracy = 100. * correct / total_samples
        
        # Log metrics to wandb
        if wandb.run:
            wandb.log({
                f'{prefix}/loss': avg_loss,
                f'{prefix}/accuracy': accuracy,
                f'{prefix}/eval_time': eval_time,
                f'{prefix}/samples': total_samples
            })
        
        print(f"\nEvaluation Summary:")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Evaluation Time: {eval_time:.2f}s")

        return accuracy, avg_loss
    
    def train_and_evaluate(
            self,
            train_loader: DataLoader,
            test_loader: DataLoader
    ) -> Tuple[List[float], List[float], float, float, List[List[torch.Tensor]]]:
        """
        Main training and evaluation function.
        
        Args:
            train_loader (torch.utils.data.DataLoader): The training loader.
            test_loader (torch.utils.data.DataLoader): The test loader.
        """
        logging.info(f"Starting training for subspace type: {self.subspace_type}")

        train_losses = []
        train_accuracies = []
        top_k_eigenvalues = []
        best_accuracy = 0.0

        for epoch in range(self.epoch):
            # Train step
            epoch_loss, epoch_accuracy, epoch_time, eigenvalues_list = self.step(
                train_loader=train_loader,
                epoch=epoch
            )
            
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)
            top_k_eigenvalues.append(eigenvalues_list)

            logging.info(f"\nEpoch {epoch} Summary:")
            logging.info(f"Average Loss: {epoch_loss:.6f}")
            logging.info(f"Accuracy: {epoch_accuracy:.2f}%")
            logging.info(f"Time taken: {epoch_time:.2f}s")

            # Evaluate on test set
            if (epoch + 1) % self.eval_freq == 0:
                eval_acc, eval_loss = self.evaluate(test_loader)
                
                if eval_acc > best_accuracy:
                    best_accuracy = eval_acc
                    self._save_checkpoint(epoch)

        final_eval_acc, final_eval_loss = self.evaluate(test_loader)
        
        return (
            train_losses, 
            train_accuracies, 
            final_eval_loss, 
            final_eval_acc, 
            top_k_eigenvalues
        )

    def _save_checkpoint(self, epoch: int) -> None:
        """
        Save the model checkpoint.

        Args:
            epoch (int): The current epoch.
        """
        try:
            save_path = self.save_dir / "models" / f"model_{epoch}.pt"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            }
            torch.save(checkpoint, save_path)
            logging.info(f"Checkpoint saved: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {str(e)}")
            raise

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a model checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info(f"Checkpoint loaded: {checkpoint_path}")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {str(e)}")
            raise
