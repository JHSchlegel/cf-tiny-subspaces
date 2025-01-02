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
from utils.metrics import compute_overlap


# =========================================================================== #
#                        Multi-task Learning Trainer                          #
# =========================================================================== #
class MTLTrainer:
    """
    Multi-task Learning Trainer implementation that supports training with 
    multiple tasks simultaneously. This class provides comprehensive training, 
    evaluation, metric tracking, and eigenvalue analysis capabilities integrated 
    with wandb logging.
    """
    
    VALID_SUBSPACE_TYPES = {"bulk", "dominant", None}

    def __init__(
        self,
        optimizer_config: Dict,
        model: nn.Module,
        criteria: Dict[str, nn.Module],
        save_dir: str,
        task_weights: Optional[Dict[str, float]] = None,
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
            criteria (Dict[str, nn.Module]): Dictionary mapping task names to loss functions
            save_dir (str): Directory path for saving checkpoints and results
            task_weights (Optional[Dict[str, float]]): Weights for each task's loss
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
        self.criteria = criteria
        self.task_weights = task_weights or {task: 1.0 for task in criteria.keys()}
        self.optimizer_config = optimizer_config
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.subspace_type = subspace_type
        self.device = device
        
        # Initialize optimizer with SubspaceSGD for eigenvalue tracking
        self.optimizer = SubspaceSGD(
            self.model,
            criteria=list(criteria.values()),  # Pass all loss functions
            **OmegaConf.to_container(optimizer_config, resolve=True)
        )

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
        self.train_metrics = {task: {"losses": [], "accuracies": []} for task in criteria.keys()}
        self.val_metrics = {task: {"losses": [], "accuracies": []} for task in criteria.keys()}
        
        # Initialize eigenvalue tracking
        self.eigenvalues = []
        self.eigenvectors = []
        self.overlaps = []
        
        # Initialize bulk/dominant subspace tracking if needed
        if self.calculate_next_top_k:
            self.eigenvalues_next_top_k = []
            self.eigenvectors_next_top_k = []
            self.overlaps_next_top_k = []
            self.overlaps_bulk = []
            self.overlaps_bulk_next_k = []

    def _train_epoch(
        self,
        train_loaders: Dict[str, DataLoader],
        epoch: int,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train the model for one epoch across all tasks simultaneously.

        Args:
            train_loaders (Dict[str, DataLoader]): Dictionary mapping task names to their DataLoaders
            epoch (int): Current epoch number

        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing metrics for each task
        """
        self.model.train()
        epoch_metrics = {task: {"loss": 0.0, "correct": 0, "total": 0} for task in train_loaders.keys()}
        start_time = time.time()

        # Create iterators for all dataloaders
        iterators = {task: iter(loader) for task, loader in train_loaders.items()}
        max_batches = max(len(loader) for loader in train_loaders.values())

        pbar = tqdm(range(max_batches), desc=f"Epoch {epoch}")
        for batch_idx in pbar:
            total_loss = 0.0
            
            # Train on a batch from each task
            for task_name, iterator in iterators.items():
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterators[task_name] = iter(train_loaders[task_name])
                    batch = next(iterators[task_name])

                # Move batch to device
                if isinstance(batch, (tuple, list)):
                    batch = [b.to(self.device) for b in batch]
                else:
                    batch = batch.to(self.device)

                # Forward pass without task name needed for model
                outputs = self.model(batch[:-1])
                loss = self.criteria[task_name](outputs, batch[-1])  # keep task name only for selecting correct loss function
                weighted_loss = loss * self.task_weights[task_name]
                total_loss += weighted_loss

                # Update metrics
                pred = outputs.argmax(dim=1, keepdim=True)
                epoch_metrics[task_name]["correct"] += pred.eq(batch[-1].view_as(pred)).sum().item()
                epoch_metrics[task_name]["total"] += batch[-1].size(0)
                epoch_metrics[task_name]["loss"] += loss.item()

            # Backward pass on combined loss
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Step with eigenvalue calculation - simplified subsampling
            self.optimizer.step(
                fp16=False,
                subspace_type=self.subspace_type
            )
            
            # Track eigenvalues and eigenvectors
            eigenvalues, eigenvectors = self.optimizer.eigenthings
            self.eigenvalues.append(eigenvalues)
            
            if batch_idx == 0 and epoch == 0:
                self.eigenvectors.append(torch.from_numpy(eigenvectors).to(self.device))
            
            # Calculate overlaps if needed
            if len(self.eigenvectors) > 0:
                current_eigenvectors = torch.from_numpy(eigenvectors).to(self.device)
                overlap = compute_overlap(
                    self.eigenvectors[-1],
                    current_eigenvectors,
                    orthogonal_complement=False
                )
                self.overlaps.append(overlap)
                
                if self.subspace_type == "bulk":
                    bulk_overlap = compute_overlap(
                        self.eigenvectors[-1],
                        current_eigenvectors,
                        orthogonal_complement=True
                    )
                    self.overlaps_bulk.append(bulk_overlap)
                    
            # Track next top-k eigenvalues if configured
            if self.calculate_next_top_k:
                eigenvalues_next_k, eigenvectors_next_k = self.optimizer.next_top_k_eigenthings
                self.eigenvalues_next_top_k.append(eigenvalues_next_k)
                
                if batch_idx == 0 and epoch == 0:
                    self.eigenvectors_next_top_k.append(
                        torch.from_numpy(eigenvectors_next_k).to(self.device)
                    )

            # Update progress bar
            if batch_idx % self.log_interval == 0:
                current_losses = {
                    task: metrics["loss"] / (batch_idx + 1) 
                    for task, metrics in epoch_metrics.items()
                }
                pbar.set_postfix(losses=current_losses)

        # Calculate epoch statistics
        epoch_time = time.time() - start_time
        final_metrics = {}
        
        for task_name, metrics in epoch_metrics.items():
            final_metrics[task_name] = {
                "loss": metrics["loss"] / max_batches,
                "accuracy": 100.0 * metrics["correct"] / metrics["total"],
                "time": epoch_time
            }

            # Log to wandb
            if wandb.run:
                # Log task metrics
                wandb.log({
                    f"train/{task_name}/loss": final_metrics[task_name]["loss"],
                    f"train/{task_name}/accuracy": final_metrics[task_name]["accuracy"],
                    f"train/{task_name}/time": epoch_time,
                    "train/epoch": epoch,
                }, commit=False)
                
                # Log eigenvalue metrics
                if len(self.eigenvalues) > 0:
                    last_eigenvalues = self.eigenvalues[-1]
                    wandb.log({
                        "train/top_eigenvalue": last_eigenvalues[0],
                        "train/eigenvalue_ratio": last_eigenvalues[0] / last_eigenvalues[-1],
                        "train/mean_overlap": np.mean(self.overlaps[-100:]) if len(self.overlaps) > 0 else 0,
                    }, commit=False)
                    
                    if self.subspace_type == "bulk":
                        wandb.log({
                            "train/mean_bulk_overlap": np.mean(self.overlaps_bulk[-100:]) if len(self.overlaps_bulk) > 0 else 0,
                        }, commit=False)

        return final_metrics

    @torch.no_grad()
    def evaluate(
        self,
        val_loaders: Dict[str, DataLoader],
        epoch: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the model on validation data for all tasks.

        Args:
            val_loaders (Dict[str, DataLoader]): Dictionary mapping task names to validation DataLoaders
            epoch (Optional[int]): Current epoch number for logging

        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing metrics for each task
        """
        self.model.eval()
        val_metrics = {}

        for task_name, loader in val_loaders.items():
            total_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(loader, desc=f"Evaluating {task_name}")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criteria[task_name](output, target)  # keep task name only for loss function
                
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

            avg_loss = total_loss / total
            accuracy = 100.0 * correct / total
            
            val_metrics[task_name] = {
                "loss": avg_loss,
                "accuracy": accuracy
            }

            # Log to wandb
            if wandb.run and epoch is not None:
                wandb.log({
                    f"val/{task_name}/loss": avg_loss,
                    f"val/{task_name}/accuracy": accuracy,
                    "val/epoch": epoch
                }, commit=False)

        return val_metrics

    def train_and_evaluate(
        self,
        train_loaders: Dict[str, DataLoader],
        val_loaders: Dict[str, DataLoader],
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Main training and evaluation loop.

        Args:
            train_loaders (Dict[str, DataLoader]): Training data loaders for each task
            val_loaders (Dict[str, DataLoader]): Validation data loaders for each task

        Returns:
            Tuple containing dictionaries of training and validation metrics
        """
        for epoch in range(self.num_epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loaders, epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Store training metrics
            for task_name, metrics in train_metrics.items():
                self.train_metrics[task_name]["losses"].append(metrics["loss"])
                self.train_metrics[task_name]["accuracies"].append(metrics["accuracy"])

            # Evaluation phase
            if (epoch + 1) % self.eval_freq == 0:
                val_metrics = self.evaluate(val_loaders, epoch)
                
                # Store validation metrics
                for task_name, metrics in val_metrics.items():
                    self.val_metrics[task_name]["losses"].append(metrics["loss"])
                    self.val_metrics[task_name]["accuracies"].append(metrics["accuracy"])

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
        """Save training, validation, and eigenvalue metrics to CSV files."""
        # Prepare training data
        train_data = []
        val_data = []

        for task_name in self.train_metrics.keys():
            for epoch in range(len(self.train_metrics[task_name]["losses"])):
                train_data.append({
                    "task": task_name,
                    "epoch": epoch,
                    "loss": self.train_metrics[task_name]["losses"][epoch],
                    "accuracy": self.train_metrics[task_name]["accuracies"][epoch]
                })

            for epoch in range(len(self.val_metrics[task_name]["losses"])):
                val_data.append({
                    "task": task_name,
                    "epoch": epoch,
                    "loss": self.val_metrics[task_name]["losses"][epoch],
                    "accuracy": self.val_metrics[task_name]["accuracies"][epoch]
                })

        # Prepare eigenvalue data
        eigenvalue_data = []
        for step, eigenvalues in enumerate(self.eigenvalues):
            for i, value in enumerate(eigenvalues):
                eigenvalue_data.append({
                    "step": step,
                    "eigenvalue_nr": i + 1,
                    "value": value
                })
        
        # Prepare overlap data
        overlap_data = []
        for step, overlap in enumerate(self.overlaps):
            overlap_data.append({
                "step": step,
                "overlap": overlap,
                "type": "dominant"
            })
            if self.subspace_type == "bulk" and step < len(self.overlaps_bulk):
                overlap_data.append({
                    "step": step,
                    "overlap": self.overlaps_bulk[step],
                    "type": "bulk"
                })

        # Save all metrics to CSV
        pd.DataFrame(train_data).to_csv(self.save_dir / "metrics/train_metrics.csv", index=False)
        pd.DataFrame(val_data).to_csv(self.save_dir / "metrics/val_metrics.csv", index=False)
        pd.DataFrame(eigenvalue_data).to_csv(self.save_dir / "metrics/eigenvalues.csv", index=False)
        pd.DataFrame(overlap_data).to_csv(self.save_dir / "metrics/overlaps.csv", index=False)

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