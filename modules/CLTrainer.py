""" 
This module contains the implementation of a Trainer class suited for training 
neural networks in a continual learning setting.
"""


# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import time
from tqdm import tqdm
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging

import os
import sys
import wandb

from utils.wandb_utils import setup_wandb
from utils.reproducibility import set_all_seeds
from utils.data_utils.continual_dataset import ContinualDataset



# =========================================================================== #
#                        Continual Learning Trainer                           #
# =========================================================================== #
class CLTrainer:
    """
    Subspace-restricted Trainer implementation in Continual Learning setting that supports
    training with externally configured optimizers. This class provides comprehensive
    training, evaluation, and metric tracking capabilities integrated with wandb logging.
    """

    VALID_SUBSPACE_TYPES = {"bulk", "dominant", None}

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        save_dir: str,
        num_tasks: int,
        num_epochs: int,
        log_interval: int,
        eval_freq: int,
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
        Initialize an instance of the Continual Learning Trainer.

        Args:
            model: Neural network model being trained
            optimizer: Custom-built subspace optimizer
            criterion: Loss function used for training
            save_dir: Directory path to save models and visualizations
            num_tasks: Number of tasks in the continual learning setting
            num_epochs: Number of epochs for training
            log_interval: Interval for logging metrics during training
            eval_freq: Frequency of evaluation on the test set
            checkpoint_freq: Frequency of saving model checkpoints
            seed: Random seed for reproducibility
            subspace_type: Type of subspace projection ("bulk" or "dominant")
            scheduler: Learning rate scheduler
            device: Device to run computations on
            use_wandb: Whether to initialize wandb logging
            wandb_log_dir: Logging directory for weights and biases
            wandb_project: Project name of weights and biases
            wandb_config: Configs used for run; saved in weights and biases
        """
        if subspace_type not in self.VALID_SUBSPACE_TYPES:
            raise ValueError(
                f"subspace_type must be one of {self.VALID_SUBSPACE_TYPES}"
            )

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
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

        self.setup_wandb: bool = use_wandb
        if self.setup_wandb:
            setup_wandb(dir=wandb_log_dir, project=wandb_project, config=wandb_config)

        logging.basicConfig(level=logging.INFO)

        self.num_tasks = num_tasks

        # initialize per task accuracies and losses for the test set:
        self.test_accuracies = {i: [] for i in range(self.num_tasks)}
        self.test_losses = {i: [] for i in range(self.num_tasks)}

    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        first_task: bool = False,
    ) -> Tuple[float, float, float, List[torch.Tensor]]:
        """
        Train the model for one epoch with or without subspace projection.

        Args:
            train_loader (DataLoader): DataLoader containing the current task's training data
            epoch (int): Current epoch number for training
            first_task (bool, optional): Whether the current task is the first task;
                if it is the first task, standard SGD will be used. Defaults to False.

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
            if batch_idx >2:
                break
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
                subspace_type=None if first_task else self.subspace_type,
            )

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
                progress_percentage = 100.0 * batch_idx / len(train_loader)

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
            last_eigenvalues = eigenvalues_list[-1]
            wandb.log(
                {
                    "train/epoch_loss": epoch_loss,
                    "train/epoch_accuracy": epoch_accuracy,
                    "train/epoch_time": epoch_time,
                    "train/epoch": epoch,
                    "train/top_eigenvalue": last_eigenvalues[0],
                    "train/eigenvalue_ratio": last_eigenvalues[0]   
                    / last_eigenvalues[-1],
                    "train/epoch": epoch,
                    },
                    commit = False # commit only once at the end of the epoch to avoid multiple steps
                )

        return epoch_loss, epoch_accuracy, epoch_time, eigenvalues_list

    @torch.no_grad()
    def _evaluate_seen_tasks(
        self,
        test_loaders: Dict[int, DataLoader],
        epoch: int=0, 
        log_to_wandb: bool = True,
        prefix: str = "eval",
    ) -> Tuple[float, float]:
        """
        Evaluate the model's performance on a test dataset and log metrics to wandb.

        Args:
            test_loaders (Dict[int, DataLoader]): Dictionary of task ids and
                test loaders for each task
            epoch (int, optional): Current epoch number for evaluation. Defaults to 0.
            log_to_wandb (bool, optional): If wandb runs, whether or not to log metrics to wandb. Defaults to True.
            prefix (str, optional): Prefix for wandb logging keys. Defaults to 'eval'.

        Returns:
            Tuple containing:
                float: Model accuracy as a percentage
                float: Average loss value across all test samples
        """
        self.model.eval()
        accuracies = {}
        avg_losses = {}
        eval_times = {}
        total_samples = {}

        for task_id, task_test_loader in test_loaders.items():
            total_loss = 0
            correct = 0
            total_samples[task_id] = 0

            # Track time for evaluation
            eval_start_time = time.time()

            pbar = tqdm(task_test_loader, desc=f"Evaluating Task {task_id}...")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                # Accumulate loss
                total_loss += loss.item() * data.size(0)

                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += torch.sum(pred.eq(target.view_as(pred))).item()
                total_samples[task_id] += data.size(0)

                # Update progress bar
                current_acc = 100.0 * correct / total_samples[task_id]
                current_loss = total_loss / total_samples[task_id]
                pbar.set_postfix(
                    {
                        f"{prefix}/loss": f"{current_loss:.4f}",
                        f"{prefix}/acc": f"{current_acc:.2f}%",
                    }
                )

            # Calculate final metrics
            eval_times[task_id] = time.time() - eval_start_time
            avg_losses[task_id] = total_loss / total_samples[task_id]
            accuracies[task_id] = 100.0 * correct / total_samples[task_id]

            # Log metrics to wandb
            if wandb.run and log_to_wandb:
                wandb.log(
                    {
                        f"{prefix}/epoch": epoch,
                        f"{prefix}/task_{task_id}_loss": avg_losses[task_id],
                        f"{prefix}/task_{task_id}_accuracy": accuracies[task_id],
                        f"{prefix}/task_{task_id}_eval_time": eval_times[task_id],
                        f"{prefix}/task_{task_id}_samples": total_samples[task_id],
                    },
                    commit = False # commit only once at the end of the epoch to avoid multiple steps
                )

            logging.info(
                f"\nEvaluation Summary: Task {task_id}\n"
                f"Accuracy: {accuracies[task_id]:.2f}%\n"
                f"Average Loss: {avg_losses[task_id]:.4f}\n"
                f"Evaluation Time: {eval_times[task_id]:.2f}s"
            )

        return accuracies, avg_losses

    def train_and_evaluate(
        self, cl_dataset: ContinualDataset
    ) -> Tuple[List[float], List[float], float, float, List[List[torch.Tensor]]]:
        """
        Main training and evaluation function.

        Args:
            cl_dataset (ContinualDataset): Continual learning dataset object that
                provides data in a sequential manner.
        """
        logging.info(f"Starting training for subspace type: {self.subspace_type}")

        train_losses = {i: [] for i in range(self.num_tasks)}
        train_accuracies = {i: [] for i in range(self.num_tasks)}
        top_k_eigenvalues = {i: [] for i in range(self.num_tasks)}

        for task_id in range(self.num_tasks):
            train_loader, test_loaders = cl_dataset.get_task_dataloaders(task_id)
            
            logging.info(f"Training on Task {task_id}...")

            for epoch in range(self.num_epochs):
                # Train step
                epoch_loss, epoch_accuracy, epoch_time, eigenvalues_list = (
                    self._train_epoch(train_loader=train_loader, epoch=epoch, first_task=task_id == 0)
                )

                train_losses[task_id].append(epoch_loss)
                train_accuracies[task_id].append(epoch_accuracy)
                top_k_eigenvalues[task_id].append(eigenvalues_list)

                logging.info(f"\nTask {task_id} Epoch {epoch} Summary:")
                logging.info(f"Average Loss: {epoch_loss:.6f}")
                logging.info(f"Accuracy: {epoch_accuracy:.2f}%")
                logging.info(f"Time taken: {epoch_time:.2f}s")

                # Evaluate on test set
                if (epoch + 1) % self.eval_freq == 0:
                    eval_acc, eval_loss = self._evaluate_seen_tasks(test_loaders, epoch)
                if (epoch + 1) % self.checkpoint_freq == 0:
                    self._save_checkpoint(epoch)
                    
                
                # commit only once at the end of the epoch to avoid multiple steps
                if wandb.run:
                    wandb.log({}, commit = True)

            test_accs_current, test_avg_losses_current = self._evaluate_seen_tasks(
                test_loaders, log_to_wandb=False,
            )
            # append the test accuracies and losses for the current task:
            for id in range(task_id + 1):
                self.test_accuracies[id].append(test_accs_current.get(id, 0))
                self.test_losses[id].append(test_avg_losses_current[id])
                
        # save to csv as backup
        self._save_metrics_to_csv(
            train_losses,
            train_accuracies,
            self.test_losses,
            self.test_accuracies,
            top_k_eigenvalues,
        )

        return (
            train_losses,
            train_accuracies,
            self.test_accuracies,
            self.test_losses,
            top_k_eigenvalues,
        )
    def _save_metrics_to_csv(
        self,
        train_losses: Dict[int, List[float]],
        train_accuracies: Dict[int, List[float]],
        test_losses: Dict[int, List[float]],
        test_accuracies: Dict[int, List[float]],
        top_k_eigenvalues: Dict[int, List[List[torch.Tensor]]],
    ) -> None:
        """
        Save training and test metrics as well as eigenvalues to CSV files.

        Args:
            train_losses (Dict[int, List[float]]): Per-task training losses
            train_accuracies (Dict[int, List[float]]): Per-task training accuracies
            test_losses (Dict[int, List[float]]): Per-task test losses
            test_accuracies (Dict[int, List[float]]): Per-task test accuracies
            top_k_eigenvalues (Dict[int, List[List[torch.Tensor]]]): Per-task top-k eigenvalues
                that were calculated during training for each batch for every epoch.
        """
        # prepare data for training metrics
        train_data = []
        test_data = []
        eigenvalue_data = []
                
        for task_id in range(self.num_tasks):
            for epoch in range(self.num_epochs):
                # training metrics
                if epoch < len(train_losses[task_id]):
                    train_data.append({
                        'task_id': task_id,
                        'epoch': epoch,
                        'loss': train_losses[task_id][epoch],
                        'accuracy': train_accuracies[task_id][epoch]
                    })
                
                # eigenvalues
                if epoch < len(top_k_eigenvalues[task_id]):
                    eigenvalues = top_k_eigenvalues[task_id][epoch]
                    for batch_idx, eigen_value in enumerate(eigenvalues):
                        eigenvalue_data.append({
                            'task_id': task_id,
                            'epoch': epoch,
                            'batch_id': batch_idx,
                            # reverse eigenvalues to have the largest eigenvalue first
                            'value': eigen_value.item() if isinstance(eigen_value, torch.Tensor) else eigen_value[::-1]
                        })
            # test metrics
            for evaluated_at_task in range(self.num_tasks):
                if evaluated_at_task >= task_id:
                    test_data.append({
                        'task_id': task_id,
                        'evaluated_at_task': evaluated_at_task,
                        'loss': test_losses[task_id][evaluated_at_task-task_id],
                        'accuracy': test_accuracies[task_id][evaluated_at_task-task_id]
                    })
                
        
        # create dataframes for saving
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        eigenvalue_df = pd.DataFrame(eigenvalue_data).explode("value")
        # add eigenvalue_nr column
        eigenvalue_df['eigenvalue_nr'] = eigenvalue_df.groupby(['task_id', 'epoch', 'batch_id']).cumcount() + 1
        eigenvalue_df = eigenvalue_df.reset_index(drop=True)
        
        # save to csv
        train_df.to_csv(
            self.save_dir / "metrics" / "train_metrics.csv",
            index=False
        )
        test_df.to_csv(
            os.path.join(self.save_dir, "metrics/test_metrics.csv"),
            index=False
        )
        eigenvalue_df.to_csv(
            os.path.join(self.save_dir, "metrics/eigenvalues.csv"),
            index=False
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
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "test_accuracies": self.test_accuracies,
                "test_losses": self.test_losses,
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
