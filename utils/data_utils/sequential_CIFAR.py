# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from .continual_dataset import ContinualDataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms, CIFAR100
from PIL import Image
from typing import List, Tuple, Dict, Optional, Iterator


class CL_CIFAR100(ContinualDataset):
    """
    Continual dataset class for the CIFAR-100 dataset
    """

    def __init__(self, num_tasks, seed:int=42)->None:
        super().__init__(num_tasks, seed)

        self.classes_per_task = 100 // num_tasks 
        self.class_order = np.arange(100)
        self.rng.shuffle(self.class_order)
    
    def setup_tasks(self, batch_size:int, data_root:str="./data", num_workers:int=4) -> None:
        """
        Create training and test dataloaders for all tasks

        Args:
            batch_size (int): Batch size.
            data_root (str, optional): Root directory of the data. Defaults to "./data".
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
        """

        MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)])
        transform_test = transforms.Compose(
                transforms.ToTensor(), 
                transforms.Normalize(MEAN, STD)
            )
        train_dataset = CIFAR100(data_root, train=True, download=True, transform=transform_train)
        test_dataset = CIFAR100(data_root, train=False, download=True, transform=transform_test)

        for task_id in range(self.num_tasks):
            start_class = task_id * self.classes_per_task
            end_class = (task_id + 1) * self.classes_per_task

            task_classes = self.class_order[start_class:end_class]

            train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in task_classes]
            test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in task_classes]

            train_subset = Subset(train_dataset, train_indices)
            test_subset = Subset(test_dataset, test_indices)

            self.train_loaders[task_id] = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
            self.test_loaders[task_id] = DataLoader(
                test_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            