"""
This module provides an abstract class for continual learning datasets.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import random


class ContinualDataset(ABC):
    """
    Abstract base class for continual learning datasets.
    """

    def __init__(
        self,
        num_tasks: int,
        seed: int = 42,
    ):
        """
        Initialize the continual learning dataset.

        Args:
            num_tasks (int): Number of tasks.
            seed (int, optional): Seed used for random componetns. Defaults to 42.
        """
        self.num_tasks = num_tasks
        self.seed = seed

        self.current_task_id = 0

        # set seeds for reproducibility:
        self.rng = np.random.default_rng(seed)

        self.train_loader = {}
        self.test_loaders = {}

    @abstractmethod
    def setup_tasks(
        self,
        batch_size: int,
        data_root: str = "../../data",
        num_workers: int = 4,
        **kwargs
    ) -> None:
        """
        Create training and test dataloaders for all tasks.

        Args:
            batch_size (int): Batch size.
            root (str, optional): Root directory of the data. Defaults to "../../data".
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
        """
        pass

    def get_task_dataloaders(
        self,
        task_id: int,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Get the training dataloader of the current task as well as the test dataloaders
        of all tasks seen so far.

        Args:
            task_id (int): Task ID.

        Returns:
            Tuple[DataLoader, Dict[int, DataLoader]]: Training dataloader for a specific task
                and a dictionary of test dataloaders for all tasks seen so far.
        """
        return (
            self.train_loader[task_id],
            {id: self.test_loaders[id] for id in range(task_id + 1)},
        )
