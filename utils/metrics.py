"""
This module includes metrics that are used in the trainer to evaluate the model's
performance and the subspace properties.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import torch


# =========================================================================== #
#                                Overlap Metric                               #
# =========================================================================== #
def compute_overlap(V_t:torch.Tensor, V_t_prime:torch.Tensor, orthogonal_complement:bool=False):
    """
    Calculate the overlap between two subspaces spanned by the orthonormal eigenbases
    V_t and V_t_prime at two different times t and t' respectively. We follow the
    definition of equation (5) in https://arxiv.org/pdf/1812.04754.
    

    Args:
        V_t (torch.Tensor): Orthonormal eigenbasis at time t. Shape: (k, num_params)
        V_t_prime (torch.Tensor): Orthonormal eigenbasis at time t'. Shape: (k, num_params)
        orthogonal_complement (bool): Whether top-k dominant subspace or it's orthonormal complement,
            the bulk subspace should be considered for overlap calculations. Defaults to False.
    """
    # project eigenvector of time t prime onto space spanned by eigenvectors of time t:
    projections = torch.mm(V_t.T, torch.mm(V_t, V_t_prime.T)) # dim out: (num_params, k)
    if orthogonal_complement:
        projections = V_t_prime.T - projections
    # ∑ᵢ ||P_top^(t)v_i^(t')||_2^2 = ||P_top^(t)V^(t')||_F^2
    overlap = torch.linalg.matrix_norm(projections, ord="fro")**2 / V_t.shape[0] # divide by k for mean
    return overlap.item()