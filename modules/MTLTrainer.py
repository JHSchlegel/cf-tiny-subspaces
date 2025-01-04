"""
This module contains the implementation of a Trainer class suited for training 
neural networks in a multitask learning setting.
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
from omegaconf import OmegaConf
import wandb

from utils.wandb_utils import setup_wandb
from utils.reproducibility import set_all_seeds
from modules.subspace_sgd import SubspaceSGD


# =========================================================================== #
#                        Multi-task Learning Trainer                          #
# =========================================================================== #
class MTLTrainer:
    """
    Multi-task Learning Trainer implementation that treats all tasks as a single
    classification problem. This class provides comprehensive training, evaluation,
    and metric tracking capabilities integrated with wandb logging.
    """
    
    VALID_SUBSPACE_TYPES = {"bulk", "dominant", None}

    def __init__(
        self,
        optimizer: SubspaceSGD,
        model: nn.Module,
        criterion: Dict[str, nn.Module],
        save_dir: str,
        num_epochs: int = 100,
        log_interval: int = 10,
        eval_freq: int = 1,
        checkpoint_freq: int = 10,
        seed: int = 42,
        subspace_type: Optional[str] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_wandb: bool = True,
        wandb_log_dir: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            optimizer_config (Dict): Configuration dictionary for the optimizer setup
            model (nn.Module): Neural network model to be trained
            criterion (Dict[str, nn.Module]): Dictionary mapping task names to loss functions
            save_dir (str): Directory path for saving checkpoints and results
            num_epochs (int): Number of training epochs
            log_interval (int): Frequency of logging training metrics (in iterations)
            eval_freq (int): Frequency of evaluation (in epochs)
            checkpoint_freq (int): Frequency of model checkpointing (in epochs).
                Defaults to 10.
            seed (int): Random seed for reproducibility. Defaults to 42.
            subspace_type (Optional[str]): Type of subspace restriction {'bulk', 'dominant', None}.
                Defaults to None.
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
        if subspace_type not in self.VALID_SUBSPACE_TYPES:
            raise ValueError(
                f"subspace_type must be one of {self.VALID_SUBSPACE_TYPES}"
            )
            
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer= optimizer
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.subspace_type = subspace_type
        self.device = device
        
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

        # Initialize metrics tracking
        self.train_metrics = {"losses": [], "accuracies": []}
        self.val_metrics = {"losses": [], "accuracies": []}
                
    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train the model for one epoch.

        Args:
            train_loader (DataLoader): DataLoader containing all training data
            epoch (int): Current epoch number

        Returns:
            Dict[str, float]: Dictionary containing metrics for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            outputs = self.model(data)
            loss = self.criterion(outputs, target)

            self.optimizer.zero_grad()
            loss.backward()
            
            self.optimizer.step(
                fp16=False,
                subspace_type=self.subspace_type
            )
            
            # Update metrics
            total_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            # Update progress bar
            if batch_idx % self.log_interval == 0:
                current_loss = total_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.2f}%'
                })

        # Calculate epoch statistics
        epoch_time = time.time() - start_time
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total

        # Log to wandb
        if wandb.run:
            wandb.log({
                "train/loss": epoch_loss,
                "train/accuracy": epoch_accuracy,
                "train/time": epoch_time,
                "train/epoch": epoch,
            }, commit=False)
                            
        return {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "time": epoch_time
        }

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        epoch: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the model on validation data.

        Args:
            val_loader (DataLoader): DataLoader containing validation data
            epoch (Optional[int]): Current epoch number for logging

        Returns:
            Dict[str, float]: Dictionary containing metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(val_loader, desc="Evaluating")
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            output = self.model(data)
            loss = self.criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            # Update progress bar
            current_loss = total_loss / total
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

        avg_loss = total_loss / total
        accuracy = 100. * correct / total

        # Log to wandb
        if wandb.run and epoch is not None:
            wandb.log({
                "val/loss": avg_loss,
                "val/accuracy": accuracy,
                "val/epoch": epoch
            }, commit=False)

        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }

    def train_and_evaluate(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Main training and evaluation loop.

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader

        Returns:
            Tuple containing training and validation metrics
        """
        for epoch in range(self.num_epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)

            # Store training metrics
            self.train_metrics["losses"].append(train_metrics["loss"])
            self.train_metrics["accuracies"].append(train_metrics["accuracy"])

            # Evaluation phase
            if (epoch + 1) % self.eval_freq == 0:
                val_metrics = self.evaluate(val_loader, epoch)
                
                # Store validation metrics
                self.val_metrics["losses"].append(val_metrics["loss"])
                self.val_metrics["accuracies"].append(val_metrics["accuracy"])

            # Save checkpoint
            if (epoch + 1) % self.checkpoint_freq == 0:
                self._save_checkpoint(epoch)

            # Commit wandb logs
            if wandb.run:
                wandb.log({}, commit=True)

        # Save final metrics
        self._save_metrics_to_csv()
        
        return self.train_metrics, self.val_metrics

    def _save_metrics_to_csv(self) -> None:
        """
        Save training and validation metrics to CSV files.

        """
        metrics_data = []
        for epoch in range(len(self.train_metrics["losses"])):
            metrics_data.append({
                "epoch": epoch,
                "train_loss": self.train_metrics["losses"][epoch],
                "train_accuracy": self.train_metrics["accuracies"][epoch],
                "val_loss": self.val_metrics["losses"][epoch] if epoch < len(self.val_metrics["losses"]) else None,
                "val_accuracy": self.val_metrics["accuracies"][epoch] if epoch < len(self.val_metrics["accuracies"]) else None
            })

        pd.DataFrame(metrics_data).to_csv(self.save_dir / "metrics/all_metrics.csv", index=False)

    def _save_checkpoint(self, epoch: int) -> None:
        """
        Save a model checkpoint.

        Args:
            epoch (int): Current epoch number
        """
        try:
            save_path = self.save_dir / "models" / f"model_{epoch}.pt"
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "train_metrics": self.train_metrics,
                "val_metrics": self.val_metrics
            }
            torch.save(checkpoint, save_path)
            logging.info(f"Checkpoint saved: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {str(e)}")
            raise

    def load_checkpoint(self, checkpoint_path: str) -> None:
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
            self.train_metrics = checkpoint["train_metrics"]
            self.val_metrics = checkpoint["val_metrics"]
            logging.info(f"Checkpoint loaded: {checkpoint_path}")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {str(e)}")
            raise