# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from .continual_dataset import ContinualDataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from PIL import Image
from typing import List, Tuple, Dict, Optional, Iterator
import warnings


class Relabel(Dataset):
    """Wrapper dataset that relabels class indices within each task"""
    def __init__(self, dataset: Dataset, task_classes: List[int]):
        self.dataset = dataset
        self.task_classes = task_classes
        # Create mapping from original class indices to new indices (0 to N-1)
        self.class_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(task_classes)}
        
    def __getitem__(self, index):
        img, label = self.dataset[index]
        # Map the original label to new label (0 to N-1)
        new_label = self.class_mapping[label]
        return img, new_label
    
    def __len__(self):
        return len(self.dataset)

class CL_CIFAR100(ContinualDataset):
    """
    Continual dataset class for the CIFAR-100 dataset
    """

    def __init__(self, classes_per_task:int=10, num_tasks:int=10, seed:int=42)->None:
        super().__init__(num_tasks, seed)

        assert num_tasks * classes_per_task <= 100, "Configuration of classes_per_task and num_tasks exceeds maximal number of classes available"
        self.classes_per_task = classes_per_task
        self.class_order = np.random.choice(100, size = num_tasks * self.classes_per_task, replace=False)

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
                [transforms.ToTensor(), 
                transforms.Normalize(MEAN, STD)]
            )
        train_dataset = datasets.CIFAR100(data_root, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(data_root, train=False, download=True, transform=transform_test)

        for task_id in range(self.num_tasks):
            start_class = task_id * self.classes_per_task
            end_class = (task_id + 1) * self.classes_per_task

            task_classes = self.class_order[start_class:end_class]

            train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in task_classes]
            test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in task_classes]

            train_subset = Subset(train_dataset, train_indices)
            test_subset = Subset(test_dataset, test_indices)

            train_relabeled = Relabel(train_subset, task_classes)
            test_relabeled = Relabel(test_subset, task_classes)

            self.train_loaders[task_id] = DataLoader(
                train_relabeled,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
            self.test_loaders[task_id] = DataLoader(
                test_relabeled,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

class CL_CIFAR10(ContinualDataset):
    """
    Continual dataset class for the CIFAR-100 dataset
    """

    def __init__(self, classes_per_task:int=2, num_tasks:int=5, seed:int=42)->None:
        super().__init__(num_tasks, seed)

        assert num_tasks * classes_per_task <= 10, "Configuration of classes_per_task and num_tasks exceeds maximal number of classes available"       
        self.classes_per_task = classes_per_task
        self.class_order = np.random.choice(10, size = num_tasks * self.classes_per_task, replace=False)
    
    def setup_tasks(self, batch_size:int, data_root:str="./data", num_workers:int=4) -> None:
        """
        Create training and test dataloaders for all tasks

        Args:
            batch_size (int): Batch size.
            data_root (str, optional): Root directory of the data. Defaults to "./data".
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
        """

        MEAN, STD = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)])
        transform_test = transforms.Compose(
                [transforms.ToTensor(), 
                transforms.Normalize(MEAN, STD)]
            )
        train_dataset = datasets.CIFAR10(data_root, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_root, train=False, download=True, transform=transform_test)

        for task_id in range(self.num_tasks):
            start_class = task_id * self.classes_per_task
            end_class = (task_id + 1) * self.classes_per_task

            task_classes = self.class_order[start_class:end_class]

            train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in task_classes]
            test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in task_classes]

            train_subset = Subset(train_dataset, train_indices)
            test_subset = Subset(test_dataset, test_indices)

            train_relabeled = Relabel(train_subset, task_classes)
            test_relabeled = Relabel(test_subset, task_classes)

            self.train_loaders[task_id] = DataLoader(
                train_relabeled,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
            self.test_loaders[task_id] = DataLoader(
                test_relabeled,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )            
            