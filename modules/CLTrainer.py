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
from utils.metrics import compute_overlap
from modules.subspace_sgd import SubspaceSGD


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
        optimizer: SubspaceSGD,
        model: nn.Module,
        criterion: nn.Module,
        save_dir: str,
        num_tasks: int,
        num_epochs: int,
        log_interval: int,
        eval_freq: int,
        task_il: bool,
        num_subsamples_Hessian: Optional[int] = 5_000,
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
            optimizer (SubspaceSGD): Configuration dictionary for the optimizer setup
            model (nn.Module): Neural network model to be trained
            criterion (nn.Module): Loss function module
            save_dir (str): Directory path for saving checkpoints and results
            num_tasks (int): Number of continual learning tasks
            num_epochs (int): Number of training epochs per task
            log_interval (int): Frequency of logging training metrics (in iterations)
            eval_freq (int): Frequency of evaluation (in epochs)
            num_subsamples_Hessian (Optional[int]): Number of samples for Hessian computation.
                Defaults to 5000.
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
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.subspace_type = subspace_type
        self.device = device
        self.task_il = task_il  # whether problem is of type task-il

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
        self.num_subsamples_Hessian = num_subsamples_Hessian

        # initialize per task accuracies and losses for the test set:
        self.test_accuracies = {i: [] for i in range(self.num_tasks)}
        self.test_losses = {i: [] for i in range(self.num_tasks)}

        self.calculate_next_top_k = self.optimizer.calculate_next_top_k

        self.eigenvectors = {i: [] for i in range(self.num_tasks - 1)}
        self.eigenvectors_next_top_k = (
            {i: [] for i in range(self.num_tasks - 1)}
            if self.calculate_next_top_k
            else None
        )
        self.overlaps = {i: [] for i in range(self.num_tasks - 1)}
        self.overlaps_next_top_k = (
            {i: [] for i in range(self.num_tasks - 1)}
            if self.calculate_next_top_k
            else None
        )
        self.overlaps_bulk = {i: [] for i in range(self.num_tasks - 1)}
        self.overlaps_bulk_next_k = (
            {i: [] for i in range(self.num_tasks - 1)}
            if self.calculate_next_top_k
            else None
        )

    def _subsample_train_loader(
        self,
        train_loader: DataLoader,
    ) -> DataLoader:
        """
        Subsample the training data loader to a smaller size for receiving more precise
        estimates of the Hessian and its eigenvalues whilst still being computationally
        tractable. Heavily inspired by Giulia Lanzillotta's implementation
        # see https://github.com/GiuliaLanzillotta/mammoth/blob/99c01d216332ec4cb3c9123a887b777410415b8a/scripts/perturbations.py#L42

        Args:
            train_loader (DataLoader): Original train dataloader.

        Returns:
            DataLoader: Dataloader with subsampled data.
        """
        # heavily inspired by Giulia Lanzillotta's implementation
        # see https://github.com/GiuliaLanzillotta/mammoth/blob/99c01d216332ec4cb3c9123a887b777410415b8a/scripts/perturbations.py#L42
        num_samples = min(len(train_loader.dataset), self.num_subsamples_Hessian)
        logging.info(
            f"Subsampling training data to {num_samples} samples for Hessian calculation"
        )

        random_indices = np.random.choice(
            list(range(len(train_loader.dataset))), size=num_samples, replace=False
        )
        subdata = torch.utils.data.Subset(train_loader.dataset, random_indices)
        logging.info(f"Finished subsampling training data")

        # return a new dataloader with the subsampled data
        return DataLoader(
            subdata,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
        )

    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        task_id: int,
        first_task: bool = False,
    ) -> Tuple[float, float, float, List[torch.Tensor]]:
        """
        Train the model for one epoch with or without subspace projection.

        Args:
            train_loader (DataLoader): DataLoader containing the current task's training data
            epoch (int): Current epoch number for training
            task_id (int): Current task id
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
        epoch_losses = []
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

            # project gradients only onto CNN layers for task-il
            if self.task_il:
                assert hasattr(self.model, "conv_layers") and hasattr(
                    self.model, "fc"
                ), "Model must have 'conv_layers' and 'fc' attributes for task-il"
                subspace_model = nn.Sequential(
                    *[self.model.conv_layers, self.model.fc[task_id]]
                )
            else:
                subspace_model = None

            # Use the custom built optimizer. We pass a standard optimization
            # in the first epoch and perform gradient projection in later steps.
            self.optimizer.step(
                data_batch=(
                    self._subsample_train_loader(train_loader)
                    if self.num_subsamples_Hessian
                    else (data, target)
                ),
                subspace_model=subspace_model,
                fp16=False,
                subspace_type=None if first_task else self.subspace_type,
            )

            eigenvalues, eigenvectors = self.optimizer.eigenthings

            if self.calculate_next_top_k:
                eigenvalues_next_top_k, eigenvectors_next_top_k = (
                    self.optimizer.next_top_k_eigenthings
                )

            eigenvalues_list.append(
                np.concatenate([eigenvalues, eigenvalues_next_top_k], axis=0)
                if self.calculate_next_top_k
                else eigenvalues
            )

            if epoch == 0 and batch_idx == 0:
                self.eigenvectors[task_id] = torch.from_numpy(eigenvectors).to(
                    self.device
                )
                if self.calculate_next_top_k:
                    self.eigenvectors_next_top_k[task_id] = torch.from_numpy(
                        eigenvectors_next_top_k
                    ).to(self.device)

            if self.calculate_next_top_k:
                for id in range(min(self.num_tasks - 1, task_id + 1)):
                    # Calculate dominant subspace overlap
                    top_k_overlap = compute_overlap(
                        self.eigenvectors[id],
                        torch.from_numpy(eigenvectors).to(self.device),
                        orthogonal_complement=False,
                    )
                    self.overlaps[id].append(top_k_overlap)

                    # Calculate bulk subspace overlap
                    bulk_overlap = compute_overlap(
                        self.eigenvectors[id],
                        torch.from_numpy(eigenvectors).to(self.device),
                        orthogonal_complement=True,
                    )

                    self.overlaps_bulk[id].append(bulk_overlap)

                    # Next top-k dominant subspace overlap
                    next_top_k_overlap = compute_overlap(
                        self.eigenvectors_next_top_k[id],
                        torch.from_numpy(eigenvectors_next_top_k).to(self.device),
                        orthogonal_complement=False,
                    )

                    self.overlaps_next_top_k[id].append(next_top_k_overlap)

                    # Next top-k bulk subspace overlap
                    next_top_k_bulk_overlap = compute_overlap(
                        self.eigenvectors_next_top_k[id],
                        torch.from_numpy(eigenvectors_next_top_k).to(self.device),
                        orthogonal_complement=True,
                    )
                    self.overlaps_bulk_next_k[id].append(next_top_k_bulk_overlap)

                    logging.info(
                        f"Overlap between top-k eigenvectors of of first step of  task {id} and current step: {top_k_overlap}"
                        f"Overlap between next top-k eigenvectors of first step of task {id} and current step: {next_top_k_overlap}"
                    )

            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            current_correct = torch.sum(pred.eq(target.view_as(pred))).item()
            correct += current_correct
            total += target.size(0)
            epoch_loss += loss.item()

            epoch_losses.append(loss.item())

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
                commit=False,  # commit only once at the end of the epoch to avoid multiple steps
            )

        return epoch_losses, epoch_accuracy, epoch_time, eigenvalues_list

    @torch.no_grad()
    def _evaluate_seen_tasks(
        self,
        test_loaders: Dict[int, DataLoader],
        epoch: int = 0,
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
            self.model._set_task(task_id)
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
                    commit=False,  # commit only once at the end of the epoch to avoid multiple steps
                )

            logging.info(
                f"\nEvaluation Summary: Task {task_id}\n"
                f"Accuracy: {accuracies[task_id]:.2f}%\n"
                f"Average Loss: {avg_losses[task_id]:.4f}\n"
                f"Evaluation Time: {eval_times[task_id]:.2f}s"
            )

        return accuracies, avg_losses

    def train_and_evaluate(self, cl_dataset: ContinualDataset) -> Tuple[
        float,
        float,
        List[float],
        List[float],
        List[float],
        List[float],
        List[List[torch.Tensor]],
    ]:
        """
        Main training and evaluation function.

        Args:
            cl_dataset (ContinualDataset): Continual learning dataset object that
                provides data in a sequential manner.
        Returns:
            Tuple containing:
                float: Average accuracy across all tasks
                float: Average maximum forgetting across all tasks
                Dict[int, List[float]]: Training losses for each task (epoch-wise)
                Dict[int, List[float]]: Training accuracies for each task (epoch-wise)
                Dict[int, List[float]]: Test accuracies for each task (epoch-wise)
                Dict[int, List[float]]: Test losses for each task (epoch-wise)
                Dict[int, List[List[torch.Tensor]]]: Top-k eigenvalues for each task (batch-wise)
        """
        logging.info(f"Starting training for subspace type: {self.subspace_type}")

        train_losses = {i: [] for i in range(self.num_tasks)}
        train_accuracies = {i: [] for i in range(self.num_tasks)}
        top_k_eigenvalues = {i: [] for i in range(self.num_tasks)}

        for task_id in range(self.num_tasks):

            # Set which task we are training
            self.model._set_task(task_id)

            train_loader, test_loaders = cl_dataset.get_task_dataloaders(task_id)

            logging.info(f"Training on Task {task_id}...")

            for epoch in range(self.num_epochs):

                # Train step
                epoch_losses, epoch_accuracy, epoch_time, eigenvalues_list = (
                    self._train_epoch(
                        train_loader=train_loader,
                        epoch=epoch,
                        first_task=task_id == 0,
                        task_id=task_id,
                    )
                )

                train_losses[task_id].append(epoch_losses)
                train_accuracies[task_id].append(epoch_accuracy)
                top_k_eigenvalues[task_id].append(eigenvalues_list)

                logging.info(f"\nTask {task_id} Epoch {epoch} Summary:")
                logging.info(f"Average Loss: {np.mean(epoch_losses):.6f}")
                logging.info(f"Accuracy: {np.mean(epoch_accuracy):.2f}%")
                logging.info(f"Time taken: {epoch_time:.2f}s")

                # Evaluate on test set
                if (epoch + 1) % self.eval_freq == 0:
                    eval_acc, eval_loss = self._evaluate_seen_tasks(test_loaders, epoch)
                if (epoch + 1) % self.checkpoint_freq == 0:
                    self._save_checkpoint(epoch)

                # commit only once at the end of the epoch to avoid multiple steps
                if wandb.run:
                    wandb.log({}, commit=True)

            test_accs_current, test_avg_losses_current = self._evaluate_seen_tasks(
                test_loaders,
                log_to_wandb=False,
            )
            # append the test accuracies and losses for the current task:
            for id in range(task_id + 1):
                self.test_accuracies[id].append(test_accs_current.get(id, 0))
                self.test_losses[id].append(test_avg_losses_current[id])

        # save to csv as backup
        average_accuracy, average_max_forgetting = self._save_metrics_to_csv(
            train_losses,
            train_accuracies,
            self.test_losses,
            self.test_accuracies,
            top_k_eigenvalues,
        )

        return (
            average_accuracy,
            average_max_forgetting,
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
        # ------------------------------------------------------------------- #
        #                       Log Metrics to CSV                            #
        # ------------------------------------------------------------------- #
        # prepare data for training metrics
        train_data = []
        test_data = []
        eigenvalue_data = []

        overlap_data = []

        for task_id in range(self.num_tasks):
            for epoch in range(self.num_epochs):
                # training metrics
                if epoch < len(train_losses[task_id]):
                    for batch_id, loss in enumerate(train_losses[task_id][epoch]):
                        train_data.append(
                            {
                                "task_id": task_id,
                                "epoch": epoch,
                                "step": batch_id,
                                "loss": loss,
                                "accuracy": train_accuracies[task_id][epoch],
                            }
                        )

                # eigenvalues
                if epoch < len(top_k_eigenvalues[task_id]):
                    eigenvalues = top_k_eigenvalues[task_id][epoch]
                    for batch_idx, eigen_value in enumerate(eigenvalues):
                        eigenvalue_data.append(
                            {
                                "task_id": task_id,
                                "epoch": epoch,
                                "batch_id": batch_idx,
                                # reverse eigenvalues to have the largest eigenvalue first
                                "value": (
                                    eigen_value.item()
                                    if isinstance(eigen_value, torch.Tensor)
                                    else eigen_value[::-1]
                                ),
                            }
                        )
            # test metrics
            for evaluated_at_task in range(self.num_tasks):
                if evaluated_at_task >= task_id:
                    test_data.append(
                        {
                            "task_id": task_id,
                            "evaluated_at_task": evaluated_at_task,
                            "loss": test_losses[task_id][evaluated_at_task - task_id],
                            "accuracy": test_accuracies[task_id][
                                evaluated_at_task - task_id
                            ],
                        }
                    )

            if self.calculate_next_top_k and task_id < self.num_tasks - 1:
                for step, (overlap, overlap_next, bulk_overlap, bulk_next) in enumerate(
                    zip(
                        self.overlaps[task_id],
                        self.overlaps_next_top_k[task_id],
                        self.overlaps_bulk[task_id],
                        self.overlaps_bulk_next_k[task_id],
                    )
                ):
                    overlap_data.append(
                        {
                            "task_id": task_id,
                            "step": step + task_id * self.num_epochs,
                            "overlap_dominant": overlap,
                            "overlap_next_top_k_dominant": overlap_next,
                            "overlap_bulk": bulk_overlap,
                            "overlap_next_top_k_bulk": bulk_next,
                        }
                    )

        # create dataframes for saving
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        eigenvalue_df = pd.DataFrame(eigenvalue_data).explode("value")

        if self.calculate_next_top_k:
            print(f"{overlap_data=}")
            overlap_df = pd.DataFrame(overlap_data)
            print(overlap_df)
            overlap_df.to_csv(self.save_dir / "metrics" / "overlaps.csv", index=False)

        # add eigenvalue_nr column
        eigenvalue_df["eigenvalue_nr"] = (
            eigenvalue_df.groupby(["task_id", "epoch", "batch_id"]).cumcount() + 1
        )
        eigenvalue_df = eigenvalue_df.reset_index(drop=True)

        # save to csv
        train_df.to_csv(self.save_dir / "metrics" / "train_metrics.csv", index=False)
        test_df.to_csv(self.save_dir / "metrics/test_metrics.csv", index=False)
        eigenvalue_df.to_csv(self.save_dir / "metrics/eigenvalues.csv", index=False)

        # ------------------------------------------------------------------- #
        #                          Average Accuracy                           #
        # ------------------------------------------------------------------- #
        # calculate average accuracy:
        avg_accuracy = test_df.query("evaluated_at_task == @self.num_tasks-1")[
            "accuracy"
        ].mean()

        # ------------------------------------------------------------------- #
        #                     Average Maximum Forgetting                      #
        # ------------------------------------------------------------------- #
        # see e.g. https://arxiv.org/pdf/2010.11635 page 3
        total_forgetting = 0.0
        # For each task j (excluding the last task)
        for j in range(self.num_tasks - 1):
            # get accuracies for task j
            task_data = test_df.query("task_id == @j")
            # get peak accuracy across all previous evaluations
            peak_accuracy = task_data["accuracy"].max()

            # get the final accuracy a_{T,j}
            final_accuracy = task_data.query("evaluated_at_task == @self.num_tasks-1")[
                "accuracy"
            ].values[0]
            # calculate forgetting for this task
            forgetting = peak_accuracy - final_accuracy
            total_forgetting += forgetting

        # Calculate average forgetting
        avg_max_forgetting = total_forgetting / (self.num_tasks - 1)

        # write to csv
        forgetting_metrics = pd.DataFrame(
            {
                "average_accuracy": [avg_accuracy],
                "average_forgetting": [avg_max_forgetting],
            }
        )

        forgetting_metrics.to_csv(
            self.save_dir / "metrics" / "forgetting_metrics.csv", index=False
        )

        return avg_accuracy, avg_max_forgetting

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
