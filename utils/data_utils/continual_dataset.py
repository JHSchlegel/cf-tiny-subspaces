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

    def __init__(self, root: str = "../../data", download: bool = True, seed: int = 42):
        """_summary_

        Args:
            root (str, optional): _description_. Defaults to "../../data".
            download (bool, optional): _description_. Defaults to True.
            seed (int, optional): _description_. Defaults to 42.
        """
        self.root = root
        self.download = download
        self.seed = seed

        # set seeds for reproducibility:
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @abstractmethod
    def get_task(self, task_id: int) -> Tuple[Dataset, Dataset]:
        """
        Get the training and test datasets for a given task.

        Args:
            task_id (int): Task ID.

        Returns:
            Tuple[Dataset, Dataset]: Training and test datasets.
        """
        pass

    @property
    @abstractmethod
    def num_tasks(self) -> int:
        """
        Get the number of tasks in the dataset.

        Returns:
            int: Number of tasks.
        """
        pass
