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
# =========================================================================== #
class CNN(nn.Module):
    """
    A simple CNN composed of two convolutional layers followed by three fully
    connected layers. The network uses ReLU activations and Max Pooling layer
    after each convolutional layer.
    """

    def __init__(
        self, 
        in_channels: int = 1,
        image_size: int = 28,
        output_dim: int = 10,
        hidden_dim: int = 128
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
        
        # Compute the size of the flattened features after convolutions
        conv_output_size = (image_size - 4) // 4 
        flatten_size = 64 * conv_output_size * conv_output_size

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(flatten_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
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
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        return x
