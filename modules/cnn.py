"""
This module provides an implementation of a simple Convolutional Neural Network (CNN)
following a similar structure to the MLP implementation.
"""

# =========================================================================== #
#                            Packages and Presets                              #
# =========================================================================== #
import torch
import torch.nn as nn


# =========================================================================== #
#                        Convolutional Neural Network                         #
#                                                                             #
# Taken from: Cohen, Kaur, Li, Zico Kolter, Talwalkar (2021, p. 34)           #
# =========================================================================== #


class CNN(nn.Module):
    """
    A simple CNN composed of two convolutional layers followed by three fully
    connected layers. The network uses ReLU activations and Max Pooling layer
    after each convolutional layer.
    """

    def __init__(
        self, 
        output_dim: int = 10,
    ):
        """
        Initialize the CNN.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 1.
            image_size (int, optional): Size of the input images. Defaults to 28.
            output_dim (int, optional): Output dimension. Defaults to 10.
            hidden_dim (int, optional): Hidden layer size. Defaults to 128.
        """
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, bias=True, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, bias=True, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, output_dim, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        
        return x
