"""
This module contains the implementation of a Trainer class suited for joint training 
of neural networks in a multi-task learning setting.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import time
from tqdm import tqdm
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from omegaconf import DictConfig, OmegaConf

import os
import sys
import wandb

from utils.wandb_utils import setup_wandb
from utils.reproducibility import set_all_seeds
from utils.data_utils.continual_dataset import ContinualDataset


# =========================================================================== #
#                          Multi-task Joint Trainer                           #
# =========================================================================== #


class JointTrainer:
    """
    Joint Training implementation that supports training with externally configured
    optimizers and handles multiple tasks simultaneously. This class provides comprehensive
    training, evaluation, and metric tracking capabilities integrated with wandb logging.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        save_dir: str,
        num_tasks: int,
        num_epochs: int,
        log_interval: int,
        eval_freq: int,
        checkpoint_freq: int = 10,
        seed: int = 42,
        cl_batch_size: int = 128,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_wandb: bool = True,
        wandb_log_dir: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            model (nn.Module): Neural network model to be trained
            criterion (nn.Module): Loss function module
            optimizer (optim.Optimizer): Optimizer instance
            save_dir (str): Directory path for saving checkpoints and results
            num_tasks (int): Number of tasks for joint training
            num_epochs (int): Number of training epochs
            log_interval (int): Frequency of logging training metrics (in iterations)
            eval_freq (int): Frequency of evaluation (in epochs)
            checkpoint_freq (int): Frequency of model checkpointing (in epochs).
                Defaults to 10.
            seed (int): Random seed for reproducibility.
                Defaults to 42.
            cl_batch_size (int): Batch size used for corresponding continual
                learning dataset. Defaults to 128.
            scheduler (Optional[_LRScheduler]): Learning rate scheduler.
                Defaults to None.
            device (str): Device to run the training on ('cuda' or 'cpu').
                Defaults to 'cuda' if available, else 'cpu'.
            use_wandb (bool): Whether to use Weights & Biases logging.
                Defaults to True.
            wandb_log_dir (Optional[str]): Directory for W&B logs.
                Defaults to None.
            wandb_project (Optional[str]): W&B project name.
                Defaults to None.
            wandb_config (Optional[Dict]): Additional configuration for W&B.
                Defaults to None.
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.device = device
        self.num_tasks = num_tasks

        # Create directories to save checkpoints
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "models").mkdir(exist_ok=True)
        (self.save_dir / "visualizations").mkdir(exist_ok=True)
        (self.save_dir / "metrics").mkdir(exist_ok=True)

        # Set seed
        set_all_seeds(seed)

        # Setup wandb
        self.setup_wandb = use_wandb
        if self.setup_wandb:
            setup_wandb(dir=wandb_log_dir, project=wandb_project, config=wandb_config)

        logging.basicConfig(level=logging.INFO)

        # Initialize per task accuracies and losses
        self.test_accuracies = {i: [] for i in range(self.num_tasks)}
        self.test_losses = {i: [] for i in range(self.num_tasks)}

        self.cl_batch_size = cl_batch_size

    def _train_epoch(
        self,
        train_loader: Dict[int, DataLoader],
        epoch: int,
    ) -> Tuple[float, float, float]:
        """
        Train the model for one epoch jointly on all tasks.

        Args:
            train_loader (Dict[int, DataLoader]): Dictionary of DataLoaders for each task
            epoch (int): Current epoch number

        Returns:
            Tuple containing:
                float: Average loss value for the epoch
                float: Accuracy percentage for the epoch
                float: Total training time for the epoch in seconds
        """
        self.model.train()
        start_time = time.time()

        # Initialize task iterators
        task_iterators = {}
        for task_id in range(self.num_tasks):
            task_iterators[task_id] = iter(train_loader[task_id])

        # Get minimum number of batches across all tasks
        num_steps = sum(
            len(loader.dataset) // self.cl_batch_size
            for loader in train_loader.values()
        )

        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(range(num_steps), desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch_idx in pbar:
            self.optimizer.zero_grad()
            batch_loss = 0.0
            batch_correct = 0
            batch_total = 0

            # Process batch from each task
            for task_id in range(self.num_tasks):
                try:
                    data, target = next(task_iterators[task_id])
                    data, target = data.to(self.device), target.to(self.device)

                    # Forward pass
                    if hasattr(self.model, "conv_layers"):
                        features = self.model.conv_layers(data)
                        outputs = self.model.fc[task_id](features)
                    elif hasattr(self.model, "features"):
                        features = self.model.features(data)
                        outputs = self.model.heads[task_id](features)
                    else:
                        outputs = self.model(data)

                    # Compute loss and accuracy
                    loss = self.criterion(outputs, target)
                    batch_loss += loss

                    # Calculate accuracy
                    pred = outputs.argmax(dim=1, keepdim=True)
                    batch_correct += pred.eq(target.view_as(pred)).sum().item()
                    batch_total += target.size(0)

                except StopIteration:
                    logging.warning(f"Task {task_id} iterator exhausted early")
                    continue
            # Backward pass and optimization
            batch_loss.backward()
            self.optimizer.step()

            # Update metrics
            epoch_loss += batch_loss.item()
            correct += batch_correct
            total += batch_total

            # Log progress
            if batch_idx % self.log_interval == 0:
                current_loss = batch_loss.item()
                current_acc = 100.0 * batch_correct / batch_total
                pbar.set_postfix(
                    {"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"}
                )

                if wandb.run:
                    wandb.log(
                        {
                            "train/batch_loss": current_loss,
                            "train/batch_accuracy": current_acc,
                            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        },
                        commit=False,
                    )

        # Calculate epoch statistics
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / (num_steps * self.num_tasks)
        avg_accuracy = 100.0 * correct / total

        return avg_loss, avg_accuracy, epoch_time

    @torch.no_grad()
    def _evaluate_tasks(
        self,
        test_loaders: Dict[int, DataLoader],
        epoch: int = 0,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Evaluate the model on validation datasets for all tasks.

        Args:
            test_loaders (Dict[int, DataLoader]): Dictionary of task ids and test loaders
            epoch (int, optional): Current epoch number for evaluation

        Returns:
            Tuple containing:
                Dict[int, float]: Dictionary of task accuracies
                Dict[int, float]: Dictionary of task losses
        """
        self.model.eval()
        accuracies = {}
        losses = {}

        for task_id in range(self.num_tasks):
            total_loss = 0
            correct = 0
            total = 0

            test_loader = test_loaders[task_id]
            pbar = tqdm(test_loader, desc=f"Evaluating Task {task_id}")

            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                if hasattr(self.model, "conv_layers"):
                    features = self.model.conv_layers(data)
                    output = self.model.fc[task_id](features)
                elif hasattr(self.model, "features"):
                    features = self.model.features(data)
                    output = self.model.heads[task_id](features)
                else:
                    output = self.model(data)

                # Compute loss
                loss = self.criterion(output, target)
                total_loss += loss.item() * data.size(0)

                # Compute accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

            # Calculate metrics
            avg_loss = total_loss / total
            accuracy = 100.0 * correct / total

            accuracies[task_id] = accuracy
            losses[task_id] = avg_loss

            if wandb.run:
                wandb.log(
                    {
                        f"eval/task_{task_id}_accuracy": accuracy,
                        f"eval/task_{task_id}_loss": avg_loss,
                        "epoch": epoch,
                    },
                    commit=False,
                )

        return accuracies, losses

    def train_and_evaluate(
        self,
        cl_dataset: ContinualDataset,
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Main training and evaluation function.

        Args:
            cl_dataset (ContinualDataset): Dataset object containing all task data

        Returns:
            Tuple containing training and evaluation metrics histories
        """
        logging.info(f"Starting joint training on {self.num_tasks} tasks")
        best_avg_accuracy = 0.0
        training_start_time = time.time()

        # Get all task dataloaders
        train_loaders = {}
        test_loaders = {}
        for task_id in range(self.num_tasks):
            train_loader, task_test_loaders = cl_dataset.get_task_dataloaders(task_id)
            train_loaders[task_id] = train_loader
            test_loaders.update(task_test_loaders)

        for epoch in range(self.num_epochs):
            # Training phase
            avg_loss, avg_accuracy, epoch_time = self._train_epoch(train_loaders, epoch)

            # Log training metrics
            logging.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
            logging.info(f"Average Loss: {avg_loss:.4f}")
            logging.info(f"Average Accuracy: {avg_accuracy:.2f}%")
            logging.info(f"Time: {epoch_time:.2f}s")

            # Periodic evaluation
            if (epoch + 1) % self.eval_freq == 0:
                accuracies, losses = self._evaluate_tasks(test_loaders, epoch)

                # Calculate average metrics
                avg_test_accuracy = sum(accuracies.values()) / len(accuracies)

                # Track best model
                if avg_test_accuracy > best_avg_accuracy:
                    best_avg_accuracy = avg_test_accuracy
                    self._save_checkpoint(epoch, is_best=True)

                # Store metrics
                for task_id in range(self.num_tasks):
                    self.test_accuracies[task_id].append(accuracies[task_id])
                    self.test_losses[task_id].append(losses[task_id])

            # Regular checkpointing
            if (epoch + 1) % self.checkpoint_freq == 0:
                self._save_checkpoint(epoch)

            # Commit wandb logs
            if wandb.run:
                wandb.log({}, commit=True)

        # Training complete
        total_time = time.time() - training_start_time
        logging.info(f"\nTraining completed in {total_time:.2f}s")
        logging.info(f"Best average accuracy: {best_avg_accuracy:.2f}%")

        # Save final metrics to CSV
        self._save_metrics_to_csv()

        return self.test_accuracies, self.test_losses

    def _save_metrics_to_csv(self) -> None:
        """Save training and evaluation metrics to CSV files."""
        # Prepare data for metrics
        test_data = []
        for task_id in range(self.num_tasks):
            for epoch in range(len(self.test_losses[task_id])):
                test_data.append(
                    {
                        "task_id": task_id,
                        "epoch": epoch,
                        "accuracy": self.test_accuracies[task_id][epoch],
                        "loss": self.test_losses[task_id][epoch],
                    }
                )

        # Create and save DataFrame
        pd.DataFrame(test_data).to_csv(
            self.save_dir / "metrics/test_metrics.csv", index=False
        )

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save a model checkpoint.

        Args:
            epoch (int): Current epoch number
            is_best (bool): Whether this is the best model so far
        """
        try:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "test_accuracies": self.test_accuracies,
                "test_losses": self.test_losses,
            }

            # Save regular checkpoint
            save_path = self.save_dir / "models" / f"model_epoch_{epoch}.pt"
            torch.save(checkpoint, save_path)

            # Save best model if applicable
            if is_best:
                best_path = self.save_dir / "models" / "model_best.pt"
                torch.save(checkpoint, best_path)
                logging.info(f"Best model saved: {best_path}")

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
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler and checkpoint["scheduler_state_dict"]:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.test_accuracies = checkpoint["test_accuracies"]
            self.test_losses = checkpoint["test_losses"]
            logging.info(f"Checkpoint loaded: {checkpoint_path}")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {str(e)}")
            raise
