"""Convolutional Neural Network (CNN) implementation for multi-task learning.

This module provides a PyTorch implementation of a CNN architecture designed for
multi-task learning scenarios. The network consists of two convolutional layers
followed by task-specific fully connected layers.

Reference:
    Cohen, Kaur, Li, Zico Kolter, Talwalkar (2021, p. 34)
"""
from typing import Optional
import torch
import torch.nn as nn


class CNN(nn.Module):
    """A CNN architecture for multi-task learning with shared convolutional layers
    and task-specific fully connected layers.

    The network architecture consists of:
        - Two convolutional layers with ReLU activation and max pooling
        - Task-specific fully connected layers for classification

    Attributes:
        feature_dim (int): Dimension of the flattened feature space after convolutions
        task (Optional[int]): Current active task ID
        num_tasks (int): Total number of tasks
        conv_layers (nn.Sequential): Shared convolutional layers
        fc (nn.ModuleList): Task-specific fully connected layers
    """

    def __init__(
        self,
        width: int = 32,
        num_tasks: int = 5, # default is CIFAR10
        classes_per_task: int = 2 # default is CIFAR10
    ) -> None:
        """Initialize the CNN architecture.

        Args:
            width (int, optional): Number of filters in convolutional layers.
                Defaults to 32.
            num_tasks (int, optional): Number of different tasks. Defaults to 5.
            classes_per_task (int, optional): Number of classes per task.
                Defaults to 2.
        """
        super(CNN, self).__init__()
        self.feature_dim = width * 64  # Spatial dimension after conv layers: 8x8 for choice of kernel
        self.task = None
        self.num_tasks = num_tasks

        # Shared convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Task-specific fully connected layers
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.feature_dim, classes_per_task)
            ) for _ in range(self.num_tasks)
        ])

    def _set_task(self, task_id: int) -> None:
        """Set the current active task.

        Args:
            task_id (int): ID of the task to be solved

        Note:
            This method also freezes all task-specific layers except the current task.
        """
        self.task = task_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, classes_per_task)

        Raises:
            RuntimeError: If task is not set before forward pass
        """
        if self.task is None:
            raise RuntimeError("Task must be set before forward pass")
        
        features = self.conv_layers(x)
        return self.fc[self.task](features)