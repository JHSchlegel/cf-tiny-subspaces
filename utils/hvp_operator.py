# This file is derived from "hvp_operator.py" from pytorch-hessian-eigenthings
# Original file: https://github.com/noahgolmant/pytorch-hessian-eigenthings/blob/master/hessian_eigenthings/hvp_operator.py
# Original authors: [Noah Golmant, Zhewei Yao, Amir Gholami, Michael Mahoney, Joseph Gonzalez]
# License: MIT (see LICENSES-hessian-eigenthings)
# Changes made:
# - replaced dataloader wihth data_source when initializing the class to allow
#       for both dataloader and tuple of current batch data
# - adapted the _prepare_grad() method to handle both dataloader and tuple of current batch data
# - added comment sections for "Packages and Presets" and
#       "Hessian Vector Product Operator" for better readability


"""
This module defines a linear operator to compute the hessian-vector product
for a given pytorch model using subsampled data.
"""
# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from typing import Callable, Union, Tuple


import torch
import torch.nn as nn
import torch.utils.data as data


import hessian_eigenthings.utils as utils

from hessian_eigenthings.operator import Operator


# =========================================================================== #
#                      Hessian Vector Product Operator                        #
# =========================================================================== #
class HVPOperator(Operator):
    """
    Use PyTorch autograd for Hessian Vec product calculation
    model:  PyTorch network to compute hessian for
    data_source: either a pytorch dataloader or a tuple of the data input and targets
    of the current batch, that we get examples from to compute grads
    loss:   Loss function to descend (e.g. F.cross_entropy)
    use_gpu: use cuda or not
    max_possible_gpu_samples: max number of examples per batch using all GPUs.
    """

    def __init__(
        self,
        model: nn.Module,
        data_source: Union[data.DataLoader, Tuple[torch.Tensor, torch.Tensor]],
        criterion: Callable[[torch.Tensor], torch.Tensor],
        use_gpu: bool = True,
        fp16: bool = False,
        full_dataset: bool = True,
        max_possible_gpu_samples: int = 256,
    ):
        size = int(sum(p.numel() for p in model.parameters()))
        super(HVPOperator, self).__init__(size)
        self.grad_vec = torch.zeros(size)
        self.model = model
        if use_gpu:
            self.model = self.model.cuda()

        self.full_dataset = full_dataset

        self.is_dataloader = isinstance(data_source, data.DataLoader)

        self.criterion = criterion
        self.use_gpu = use_gpu
        self.fp16 = fp16
        self.max_possible_gpu_samples = max_possible_gpu_samples

        assert self.is_dataloader or isinstance(
            data_source, (tuple, list)
        ), "data_source must be of type tuple (list also works) or torch.utils.data.DataLoader"
        if self.is_dataloader:
            self.dataloader = data_source
            self.dataloader_iter = iter(data_source)
            if not hasattr(self.dataloader, "__len__") and self.full_dataset:
                raise ValueError(
                    "For full-dataset averaging, dataloader must have '__len__'"
                )
        else:
            self.batch_inputs, self.batch_targets = data_source
            # check whether input and target have the same number of samples:
            assert self.batch_inputs.size(0) == self.batch_targets.size(
                0
            ), "Input and target must have the same number of samples"
            self.full_dataset = False  # Override full_dataset when using single batch

    def apply(self, vec: torch.Tensor):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        if self.full_dataset:
            return self._apply_full(vec)
        else:
            return self._apply_batch(vec)

    def _apply_batch(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hessian-vector product for a mini-batch from the dataset.
        """
        # compute original gradient, tracking computation graph
        self._zero_grad()
        grad_vec = self._prepare_grad()
        self._zero_grad()
        # take the second gradient
        # this is the derivative of <grad_vec, v> where <,> is an inner product.
        hessian_vec_prod_dict = torch.autograd.grad(
            grad_vec, self.model.parameters(), grad_outputs=vec, only_inputs=True
        )
        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat(
            [g.contiguous().view(-1) for g in hessian_vec_prod_dict]
        )
        hessian_vec_prod = utils.maybe_fp16(hessian_vec_prod, self.fp16)
        return hessian_vec_prod

    def _apply_full(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hessian-vector product averaged over all batches in the dataset.

        """
        n = len(self.dataloader)
        hessian_vec_prod = None
        for _ in range(n):
            if hessian_vec_prod is not None:
                hessian_vec_prod += self._apply_batch(vec)
            else:
                hessian_vec_prod = self._apply_batch(vec)
        hessian_vec_prod = hessian_vec_prod / n
        return hessian_vec_prod

    def _zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def _prepare_grad(self) -> torch.Tensor:
        """
        Compute gradient w.r.t loss over all parameters and vectorize
        """
        # get the next batch of data or use the current batch when not using a dataloader
        if self.is_dataloader:
            try:
                all_inputs, all_targets = next(self.dataloader_iter)
            except StopIteration:
                self.dataloader_iter = iter(self.dataloader)
                all_inputs, all_targets = next(self.dataloader_iter)
        else:
            all_inputs, all_targets = self.batch_inputs, self.batch_targets

        num_chunks = max(1, len(all_inputs) // self.max_possible_gpu_samples)

        grad_vec = None

        # This will do the "gradient chunking trick" to create micro-batches
        # when the batch size is larger than what will fit in memory.
        # WARNING: this may interact poorly with batch normalization.

        input_microbatches = all_inputs.chunk(num_chunks)
        target_microbatches = all_targets.chunk(num_chunks)
        for input, target in zip(input_microbatches, target_microbatches):
            if self.use_gpu:
                input = input.cuda()
                target = target.cuda()

            output = self.model(input)
            loss = self.criterion(output, target)
            grad_dict = torch.autograd.grad(
                loss, self.model.parameters(), create_graph=True
            )
            if grad_vec is not None:
                grad_vec += torch.cat([g.contiguous().view(-1) for g in grad_dict])
            else:
                grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict])
            grad_vec = utils.maybe_fp16(grad_vec, self.fp16)
        grad_vec /= num_chunks
        self.grad_vec = grad_vec
        return self.grad_vec
