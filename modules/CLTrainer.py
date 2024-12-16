import torch
import torch.nn as nn
import numpy as np
import wandb
import time
import logging
from typing import List, Tuple, Optional, Dict, Union
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader

from utils.wandb_utils import setup_wandb



class CLTrainer:
    """
    Subspace-restricted Trainer implementation in Continual Learning setting that supports 
    training with externally configured optimizers. This class provides comprehensive
    training, evaluation, and metric tracking capabilities integrated with wandb logging.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.modules.loss._Loss,
        epoch: int,
        log_interval: int,
        subspace_type: str,
        setup_wandb: bool,        
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize an instance of the Continual Learning Trainer.

        Args:
            model: Neural network model being trained
            optimizer: Our custom-built subspace optimizer
            criterion: Loss function used for training
            epoch: Number of epochs for training
            log_interval: Interval for logging metrics
            subspace_type: Type of subspace projection ("bulk" or "dominant" or None)
            setup_wandb: Whether to initialize wandb logging
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch = epoch
        self.log_interval = log_interval
        self.subspace_type = subspace_type
        self.device = device
                
        if setup_wandb:
            setup_wandb()

    def step(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Tuple[float, float, float, List[torch.Tensor]]:
        """
        Train the model for one epoch with or without subspace projection.

        Args:
            train_loader: DataLoader for current task's training data
            epoch: Current epoch number

        Returns:
            Tuple containing:
                - float: Average epoch loss
                - float: Epoch accuracy (percentage)
                - float: Epoch training time
                - List[torch.Tensor]: List of eigenvalues during training
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

    def evaluate(
        self,
        test_loader: DataLoader,
        prefix: str = 'eval',
    ) -> Tuple[float, float]:
        """
        Perform the evaluation step of the model, together with logging
        the performance metrics to wandb dashboard.
        
        Args:
            test_loader: DataLoader for test data
            prefix: Prefix for wandb logging keys (e.g., 'eval', 'test')
        
        Returns:
            Tuple containing:
                - float: Average accuracy
                - float: Average loss
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        # Track time for evaluation
        eval_start_time = time.time()

        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Evaluating')
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