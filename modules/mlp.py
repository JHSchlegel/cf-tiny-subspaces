"""
This module provides an implementation of a simple Multi-Layer Perceptron (MLP)
inspired by the MLP class of the mammoth github repository, with the original
MLP class being available here:
https://github.com/aimagelab/mammoth/blob/3d6fc4b0645734e5d8c416293efadc44f3032382/backbone/MNISTMLP.py#L13
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import torch.nn as nn
import torch


# =========================================================================== #
#                            Multi-Layer Perceptron                           #
# =========================================================================== #
class MLP(nn.Module):
    """
    MLP composed of two hidden layers, each containing 100 ReLU activations. Inspired
    by the MLP class of the mammoth github repository:
    https://github.com/aimagelab/mammoth/blob/3d6fc4b0645734e5d8c416293efadc44f3032382/backbone/MNISTMLP.py#L13
    """

    def __init__(
        self, input_dim: int = 784, output_dim: int = 10, hidden_dim: int = 100
    ):
        """
        Initialize the MLP.

        Args:
            input_dim (int, optional): Input dimension. Defaults to 784.
            hidden_dim (int, optional): Hidden layer size. Defaults to 100.
            output_dim (int, optional): Output dimension. Defaults to 10.
        """
        super(MLP, self).__init__()

        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x.view(x.size(0), -1)
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.output(x)
        return x
