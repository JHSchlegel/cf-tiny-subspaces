"""
This module provides an implementation of the Permuted MNIST dataset. The 
PermutationTransform class is inspired by the Permutation class of the mammoth
github repository, with the original Permutation class being available here:
https://github.com/aimagelab/mammoth/blob/master/datasets/transforms/permutation.py#L9
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from .continual_dataset import ContinualDataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict, Optional, Iterator


# =========================================================================== #
#                         Permutation Transformation                          #
# =========================================================================== #
class PermutationTransform:
    """
    Custom transformation to permute the pixels of an image. Inspired by the
    Permutation class of the mammoth github repository:
    https://github.com/aimagelab/mammoth/blob/master/datasets/transforms/permutation.py#L9
    """

    def __init__(self, permutation: np.ndarray) -> None:
        self.permutation = permutation

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """GEt the permuted image.

        Args:
            x (torch.Tensor): Input image.

        Returns:
            torch.Tensor: Permuted image.
        """
        return x.view(-1)[self.permutation].view(1, 28, 28)



# =========================================================================== #
#                          Permuted MNIST Dataset                             #
# =========================================================================== #
class PermutedMNIST(ContinualDataset):
    """
    Continual dataset class for the permuted MNIST dataset.
    """

    def __init__(self, num_tasks, seed=42):
        super().__init__(num_tasks, seed)

    def setup_tasks(self, batch_size, data_root="./data", num_workers=4) -> None:
        """
        Create training and test dataloaders for all tasks.

        Args:
            batch_size (int): Batch size.
            data_root (str, optional): Root directory of the data. Defaults to "./data".
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
        """

        permutations = self._get_permutations()

        # assert whether not all permutations are the same:
        for task_id in range(self.num_tasks):
            perm = permutations[task_id]

            # Define the transformations:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    PermutationTransform(perm),
                    # need to use custom class as lambda
                    # function would just save reference to the permutation creating
                    # num_tasks-1 times the same datasets. could use lambda function
                    # with multiple arguments but would make it more complicated.
                ]
            )

            # Load the MNIST dataset with permuted pixels:
            train_dataset = datasets.MNIST(
                data_root, train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST(
                data_root, train=False, download=True, transform=transform
            )

            self.train_loaders[task_id] = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )

            self.test_loaders[task_id] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

        # assert whether the data loaders are created correctly:
        # the permuted images different for each test task but the targets are the same:
        data = np.zeros((batch_size, 28 * 28, self.num_tasks))
        target = np.zeros((batch_size, self.num_tasks))
        for task_id in range(self.num_tasks):
            test_loader = self.test_loaders[task_id]
            data_task, target_task = next(iter(test_loader))
            data[:, :, task_id] = data_task.view(batch_size, -1).numpy()
            target[:, task_id] = target_task.numpy()

            for i in range(task_id):
                assert not np.allclose(
                    data[:, :, i], data[:, :, task_id]
                ), "Images don't differ between tasks (in the test loader)"
                assert np.allclose(
                    target[:, i], target[:, task_id]
                ), "Targets differ between tasks (in the test loader)"

    def _get_permutations(self) -> List[np.ndarray]:
        """
        Generate the permutations for the dataset.

        Returns:
            List[np.ndarray]: List of permutations.
        """
        # MNIST imges have resolution 28x28
        return [self.rng.permutation(28 * 28) for _ in range(self.num_tasks)]
