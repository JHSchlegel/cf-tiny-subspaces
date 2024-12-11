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
from typing import List, Optional, Tuple
import torch.nn as nn
from numpy import np.ndarray
import numpy as np

from hessian_eigenthings.lanczos import lanczos

# =========================================================================== #
#                            Test for Orthonormality                          #
# =========================================================================== #
# function for sanity checks for calculated eigenvectors before projecting gradients
def is_orthonormal_basis(matrix: np.ndarray, device: torch.device, tol:float=1e-6) -> bool:
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
    matrix =torch.from_numpy(matrix).float().to(device)
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
    """

    def __init__(
        self,
        model: nn.Module,
        k: int,
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

        num_params = sum(p.numel() for p in model.parameters())
        # device used for creating new tensors
        self.device = next(model.parameters()).device
        # reserve memory for eigenvectors and eigenvalues
        self.eigenvectors = torch.zeros(
            (k, num_params), dtype=torch.float32, device=self.device
        )
        self.eigenvalues = numpy.zeros(k, dtype=np.float32)
        self.model = model
        # number of top-k eigenvectors to compute
        self.k = k
        # maximum number of Lanczos steps for calculation of the eigenbasis
        self.max_lanczos_steps = max_lanczos_steps
        
    @property
    def eigenthings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current eigenvalues and eigenvectors.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Retruns current eigenvalues and eigenvectors
                as separate numpy arrays.
        """
        return self.eigenvalues, self.eigenvectors.cpu().numpy()
    
    # this method was inspired by the gather_flat_grad() of the LBFGS optimizer
    # in the PyTorch source code; see: https://github.com/pytorch/pytorch/blob/main/torch/optim/lbfgs.py
    def _flatten_grad(self) -> torch.Tensor:
        """
        Flatten and concatenate gradients from all parameters. This prepares
        the gradients for projection onto subspaces through matrix-vector products.

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
        to their original shape.

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
        self, flat_grad: torch.Tensor, subspace_type: Optional[str] = None
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
            self.eigenvectors.T, torch.mv(self.eigenvectors, flat_grad)
        )

        if subspace_type == "dominant":
            return projections
        elif subspace_type == "bulk":
            return flat_grad - projections

    def _update_eigenvectors(
        self,
        data_batch: Tuple[torch.Tensor, torch.Tensor],
        fp16: bool = False,
    ) -> None:
        """
        Update the top-k eigenvectors of the Hessian using the given/current data batch.

        Args:
            data_batch: Tuple of input and target tensors
            fp16: Whether to use FP16 precision for computation
        """
        use_gpu = True if self.device.type == "cuda" else False
        hvp_operator = HVPOperator(
            self.model,
            data_batch,
            criterion,
            use_gpu=use_gpu,
            full_dataset=False,
        )
        self.eigenvalues, self.eigenvectors = lanczos(
            hvp_operator,
            self.k,
            use_gpu=use_gpu,
            fp16=fp16,
            max_steps=self.max_lanczos_steps,
        )
        
        assert is_orthonormal_basis(
            matrix = self.eigenvectors, device = self.device, tol=1e-4
        ), "Eigenvectors are not orthonormal"
        self.eigenvectors = torch.from_numpy(self.eigenvectors).to(self.device)

    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
        data_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        fp16: bool = False,
        subspace_type: Optional[str] = None,
    ) -> Optional[float]:
        """
        Performs a single SGD optimization step with gradient projected onto a
        subspace as specified in 'subspace_type'.
        
        Args:
            closure (Optional[Callable[[], float]], optional): _description_. Defaults to None.
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
        
        # if not standard SGD, project gradient onto subspace
        if subspace_type:

            # Flatten gradients to allow for matrix-vector products
            flat_grad = self._flatten_grad()
            
            # Update eigenvectors/ eigenbasis that will be used for projection
            self._update_eigenvectors(data_batch, fp16=fp16)

            # Project gradient
            projected_grad = self._project_gradient(
                flat_grad, subspace_type=subspace_type
            )

            # Unflatten and assign projected gradient back to parameters
            # i.e. update current gradients with projected gradients
            # before applying SGD step using the updated, projected parameter gradients
            self._unflatten_grad(projected_grad)

        with torch.no_grad():
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            # Perform regular SGD step with the projected gradient
            super().step(closure=closure)

            return loss