# References Disclaimer:
# The _flatten_grad() and _unflatten_grad() methods were inspired by the gather_flat_grad()
# and _add_grad() methods in the source code of the PyTorch LBFGS optimizer class
# which can be accessed here: https://github.com/pytorch/pytorch/blob/main/torch/optim/lbfgs.py


"""
This module includes an implementation of the SubspaceSGD optimizer 
that offers the possiblity to project gradients onto a subspace at every 
parameter update step. For an example usage see the subspace_sgd_examples.ipynb
notebook in the notebooks folder.
"""


# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import torch
from torch.optim.sgd import SGD
from typing import List, Optional, Tuple, Callable
import torch.nn as nn
import numpy as np
import os
import sys


from pytorch_hessian_eigenthings.hessian_eigenthings.lanczos import lanczos
from pytorch_hessian_eigenthings.hessian_eigenthings.hvp_operator import HVPOperator


# =========================================================================== #
#                            Test for Orthonormality                          #
# =========================================================================== #
# function for sanity checks for calculated eigenvectors before projecting gradients
def is_orthonormal_basis(
    matrix: np.ndarray, device: torch.device, tol: float = 1e-6
) -> bool:
    """
    Check if the rows of a matrix form an orthonormal basis.

    Args:
        matrix (np.ndarray): 2D array of shape (k, num_params) i.e. want to check
            if rows span an orthonormal basis. Hereby, k is the number of basis vectors
            and num_params is the dimensionality of the parameter space.
        torch.device (torch.device): Device on which the computation is performed.
        tol (float, optional): Tolerance for checking orthonormality. Defaults to 1e-6.

    Returns:
        bool: True if the matrix forms an orthonormal basis, False otherwise.
    """
    # since matrix M has shape (k, num_params) we need to calculate M * M^T
    # instead of typically M^T * M to check for orthonormality
    matrix = torch.from_numpy(matrix).float().to(device)
    product_matrix = torch.mm(matrix, matrix.T)

    # Check if the dot product matrix is close to the identity matrix
    identity = torch.eye(matrix.size(0), dtype=matrix.dtype, device=device)
    return torch.allclose(product_matrix, identity, atol=tol)


# =========================================================================== #
#                            Subspace SGD Optimizer                           #
# =========================================================================== #
class SubspaceSGD(SGD):
    """
    SGD optimizer that can project gradients onto the orthonormal space
    spanned by the top-k eigenvectors of the Hessian or its orthogonal complement.
    The _flatten_grad() and _unflatten_grad() methods were inspired by the gather_flat_grad()
    and _add_grad() methods in the source code of the PyTorch LBFGS optimizer class
    which can be accessed here: https://github.com/pytorch/pytorch/blob/main/torch/optim/lbfgs.py
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.modules.loss._Loss,
        k: int,
        calculate_next_top_k: bool = False,
        max_lanczos_steps: int = 50,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        """
        Initialize an instance of the SubspaceSGD optimizer that inherits from
        the PyTorch SGD optimizer.

        Args:
            model (nn.Module): Model whose parameters are optimized.
            k (int): Number of top-k eigenvalues and eigenvectors of the Hessian
                to compute.
            calculate_next_top_k (bool, optional): Whether to calculate the top-k and next top-k
                eigenvalues and eigenvectors (i.e. 2k in total) or only the
                top-k. Defaults to False.
            criterion (nn.modules.loss._Loss): Loss function that is used for
                calculation of the Hessian.
            max_lanczos_steps (int, optional): Maximum number of steps allowed in
                Lanczos' method for computing the top-k eigenvectors and eigenvalues.
                Note that if there was no convergence (up to some tolerance)
                within max_lanczos_steps, an error will be raised. Defaults to 50.
            lr (float, optional): Learning rate that will be used in the
                SGD optimizer for the gradient updates. Defaults to 1e-3.
            momentum (float, optional): SGD Momentum factor. Defaults to 0.0.
            dampening (float, optional): SGD dampening for momentum. Defaults to 0.0.
            weight_decay (float, optional): (L2) weight decay used in SGD. Defaults to 0.0.
            nesterov (bool, optional): Whether Nesterov momentum should be enabled.
                Defaults to False.
        """

        # initialize SGD optimizer
        super().__init__(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

        self.criterion = criterion  # loss function
        num_params = sum(p.numel() for p in model.parameters())
        # device used for creating new tensors
        self.device = next(model.parameters()).device

        self.calculate_next_top_k = calculate_next_top_k
        # number of top-k eigenvectors to compute
        self.k = k

        # reserve memory for eigenvectors and eigenvalues
        self.eigenvectors = torch.zeros(
            (k, num_params), dtype=torch.float32, device=self.device
        )

        self.eigenvectors_next_top_k = (
            torch.zeros((k, num_params), dtype=torch.float32, device=self.device)
            if calculate_next_top_k
            else None
        )

        self.eigenvalues = np.zeros(k, dtype=np.float32)
        self.eigenvalues_next_top_k = (
            np.zeros(k, dtype=np.float32) if calculate_next_top_k else None
        )
        self.model = model
        # maximum number of Lanczos steps for calculation of the eigenbasis
        self.max_lanczos_steps = max_lanczos_steps

        if hasattr(self.model, "conv_layers"):
            self.conv_param_size = self.conv_param_size = sum(
                p.numel() for p in self.model.conv_layers.parameters()
            )

    @property
    def eigenthings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current eigenvalues and eigenvectors.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Retruns current eigenvalues and eigenvectors
                as separate numpy arrays.
        """
        return self.eigenvalues, self.eigenvectors.cpu().numpy()

    @property
    def next_top_k_eigenthings(self) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Returns:
            Tuple[np.ndarray, np.ndarray]: Returns next top-k eigenvalues and eigenvectors
                as separate numpy arrays.
        """
        assert (
            self.calculate_next_top_k
        ), "Next top-k eigenthings were not calculated as calculate_next_top_k was set to False"
        assert (
            self.eigenvectors_next_top_k.shape[0] == self.k
        ), "The next top-k eigenvectors consist of more than k eigenvectors"

        return self.eigenvalues_next_top_k, self.eigenvectors_next_top_k

    # this method was inspired by the gather_flat_grad() of the LBFGS optimizer
    # in the PyTorch source code; see: https://github.com/pytorch/pytorch/blob/main/torch/optim/lbfgs.py
    def _flatten_grad(self) -> torch.Tensor:
        """
        Flatten and concatenate gradients from all parameters. This prepares
        the gradients for projection onto subspaces through matrix-vector products.
        this method was inspired by the gather_flat_grad() of the LBFGS optimizer
        in the PyTorch source code; see: https://github.com/pytorch/pytorch/blob/main/torch/optim/lbfgs.py

        Returns:
            torch.Tensor: Flattened gradient tensor of all parameters
        """
        grad_list = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    # append flattened gradients of parameter group to list
                    grad_list.append(p.grad.data.view(-1))

        return torch.cat(grad_list)

    # this method was inspired by the _add_grad() of the LBFGS optimizer
    # in the PyTorch source code; see: https://github.com/pytorch/pytorch/blob/main/torch/optim/lbfgs.py
    def _unflatten_grad(self, flat_grad: torch.Tensor) -> None:
        """
        Replace parameter gradients by gradient data from the (projected)
        flattened gradient tensor. Finally, reshape the parameter gradients
        to their original shape. This method was inspired by the _add_grad() of the LBFGS optimizer
        in the PyTorch source code; see: https://github.com/pytorch/pytorch/blob/main/torch/optim/lbfgs.py

        Args:
            flat_grad (torch.Tensor): tensor of flattened (and possibly projected) gradients
        """
        pointer = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    num_param = p.numel()
                    p.grad.data = flat_grad[pointer : pointer + num_param].view_as(
                        p.data
                    )
                    pointer += num_param

    def _project_gradient(
        self,
        flat_grad: torch.Tensor,
        subspace_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Project gradient onto the space spanned by the top-k eigenvectors or
        its orthogonal complement.

        Args:
            flat_grad (torch.Tensor): Flattened gradient tensor
            subspace_type (Optional[str], optional): Type of subspace to
                project onto (None or "dominant" or "bulk"). Defaults to None.

        Returns:
            torch.Tensor: Projected gradient tensor
        """

        # Calculate projections of gradient onto eigenbasis
        projections = torch.mv(
            self.eigenvectors.T,
            torch.mv(self.eigenvectors, flat_grad[: self.eigenvectors.shape[1]]),
        )

        if subspace_type == "dominant":
            return projections
        elif subspace_type == "bulk":
            return flat_grad[: self.eigenvectors.shape[1]] - projections

    def _update_eigenvectors(
        self,
        data_batch: Tuple[torch.Tensor, torch.Tensor],
        subspace_model: Optional[nn.Module] = None,
        fp16: bool = False,
    ) -> None:
        """
        Update the top-k eigenvectors of the Hessian using the given/current data batch.

        Args:
            data_batch (Tuple[torch.Tensor, torch.Tensor]): Tuple of input and target tensors
            subspace_model (Optional[nn.Module], optional): Model whose parameters are used for the
                subspace projection. If None, use self.model, i.e., the full model. Defaults to None.
            fp16 (bool, optional): Whether to use half precision for the eigenthings
        """
        use_gpu = True if self.device.type == "cuda" else False

        hvp_operator = HVPOperator(
            model=subspace_model if subspace_model else self.model,
            data_source=data_batch,
            criterion=self.criterion,
            use_gpu=use_gpu,
            full_dataset=False,
            max_possible_gpu_samples=2**16,
        )
        eigenvalues, eigenvectors = lanczos(
            operator=hvp_operator,
            num_eigenthings=2 * self.k if self.calculate_next_top_k else self.k,
            use_gpu=use_gpu,
            fp16=fp16,
            max_steps=self.max_lanczos_steps,
        )

        assert is_orthonormal_basis(
            matrix=eigenvectors, device=self.device, tol=1e-4
        ), "Eigenvectors are not orthonormal"

        if hasattr(self.model, "conv_layers") and subspace_model:
            # Trim eigenvectors to only include conv layer parameters
            eigenvectors = eigenvectors[:, : self.conv_param_size]

        self.eigenvalues = (
            eigenvalues[self.k :] if self.calculate_next_top_k else eigenvalues
        )
        self.eigenvectors = (
            eigenvectors[self.k :] if self.calculate_next_top_k else eigenvectors
        )

        self.eigenvalues_next_top_k = (
            eigenvalues[: self.k] if self.calculate_next_top_k else None
        )
        self.eigenvectors_next_top_k = (
            eigenvectors[: self.k] if self.calculate_next_top_k else None
        )

        self.eigenvectors = torch.from_numpy(self.eigenvectors).to(self.device)

    def step(
        self,
        subspace_model: Optional[nn.Module] = None,
        closure: Optional[Callable[[], float]] = None,
        data_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        fp16: bool = False,
        subspace_type: Optional[str] = None,
    ) -> Optional[float]:
        """
        Performs a single SGD optimization step with gradient projected onto a
        subspace as specified in 'subspace_type'.

        Args:
            subspace_model (Optional[nn.Module], optional): Model whose parameters are used for th 
                subspace projection. If None, use self.model, i.e., the full model. Defaults to None.
            closure (Optional[Callable[[], float]], optional): Closure argument to make step 
                function consistent with step function of other Pytorch optimizers. Defaults to None.
            data_batch (Optional[Tuple[torch.Tensor, torch.Tensor]], optional):
                Tuple of input and target tensors of the current batch.
                Is used for calcuation of the Hessian and its eigenthings.
                Defaults to None.
            fp16 (bool, optional): _description_. Defaults to False.
            subspace_type (Optional[str], optional): Type of subspace projection
                that should be used. Has to be in [None, 'dominant', 'bulk'], where
                None corresponds to standard SGD, 'dominant' to Dom-SGD and
                'bulk' to Bulk-SGD. Defaults to None.

        Returns:
            Optional[float]: Returns the loss if a closure is provided, None otherwise.
        """
        assert subspace_type in [
            None,
            "dominant",
            "bulk",
        ], "Invalid subspace type, should be None, 'dominant' or 'bulk'"

        # flatten gradients to allow for matrix-vector products;
        # this way eigenvalue calculations don't interfere with parameter gradients
        flat_grad = self._flatten_grad()

        # Update eigenvectors/ eigenbasis that will be used for projection
        # uses zero_grad internally, hence cannot just interchange order of
        # flat_grad and update_eigenvectors
        self._update_eigenvectors(data_batch, fp16=fp16, subspace_model=subspace_model)

        # if not standard/vanilla SGD, project gradient onto subspace
        if subspace_type:
            if hasattr(self, "conv_param_size") and subspace_model:
                # Project gradient
                flat_grad[: self.conv_param_size] = self._project_gradient(
                    flat_grad, subspace_type=subspace_type
                )
            else:
                flat_grad = self._project_gradient(
                    flat_grad, subspace_type=subspace_type
                )

        # Unflatten and assign projected gradient back to parameters
        # i.e. update current gradients with projected gradients
        # before applying SGD step using the updated, projected parameter gradients
        self._unflatten_grad(flat_grad)

        with torch.no_grad():
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            # Perform regular SGD step with the projected gradient
            super().step(closure=closure)

            return loss
