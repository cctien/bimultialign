from typing import Dict, Optional, Tuple, Union

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import tiny_value_of_dtype

import torch

_emb_dim = -1


def get_unit_normalized_vector(vec: Optional[torch.FloatTensor],
                               ) -> Optional[torch.FloatTensor]:
    """
    This function return the batch of vectors `vec` which are normalized to unit vectors, 
    return `None` if `vec` being `None`.

    # Input

    vec : `torch.FloatTensor`
        Shape: (batch_size, vector_size,)
    """
    if vec is None:
        return None

    return torch.true_divide(vec,
                             (torch.norm(vec, dim=-1, keepdim=True)
                              + tiny_value_of_dtype(vec.dtype)))


def get_argmax_ind(scores: torch.Tensor,
                   dim: int,
                   mask: Optional[torch.BoolTensor] = None,
                   ) -> Tuple[torch.IntTensor]:

    scores = scores.detach()
    if mask is not None:
        scores = scores.masked_fill(torch.logical_not(mask), float('-inf'))

    ind_argmax = torch.argmax(input=scores, dim=dim, keepdim=False)
    ind_range = torch.arange(end=scores.shape[1-dim])

    if dim == 1:
        return ind_range, ind_argmax
    elif dim == 0:
        return ind_argmax, ind_range
    else:
        raise NotImplementedError


def get_argmin_ind(scores: torch.Tensor,
                   dim: int,
                   mask: Optional[torch.BoolTensor] = None,
                   ) -> Tuple[torch.IntTensor]:

    scores = scores.detach()
    if mask is not None:
        scores = scores.masked_fill(torch.logical_not(mask), float('inf'))

    ind_argmin = torch.argmin(input=scores, dim=dim, keepdim=False)
    ind_range = torch.arange(end=scores.shape[1-dim])

    if dim == 1:
        return ind_range, ind_argmin
    elif dim == 0:
        return ind_argmin, ind_range
    else:
        raise NotImplementedError


def get_diag_mask(size: int,
                  stack_size: int = 1,
                  device: Optional[torch.device] = None,
                  ) -> torch.BoolTensor:

    diagonal_mask = torch.eye(size, dtype=torch.bool, device=device)
    if stack_size == 1:
        return diagonal_mask
    elif stack_size > 1:
        return diagonal_mask.expand(stack_size, size, size)
    else:
        raise ValueError(
            f"Stack size must be a positive integer. Got {stack_size} instead.")


def get_two_slices(embeddings: torch.FloatTensor,
                   total_dim: int,
                   second_dim: int = 0,
                   ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:

    if second_dim == 0:
        return embeddings, None
    else:
        first_slice = torch.narrow(embeddings,
                                   _emb_dim,
                                   0, total_dim - second_dim)

        second_slice = torch.narrow(embeddings,
                                    _emb_dim,
                                    total_dim - second_dim, second_dim)

        return first_slice, second_slice


# _layer_dim = -2

# def express_pair_relationship(vec1: torch.Tensor,
#                               vec2: torch.Tensor) -> torch.Tensor:

#     difference = vec1 - vec2
#     hadamard_product = vec1 * vec2
#     relationship = torch.cat(
#         (vec1, vec2, difference, hadamard_product), dim=-1)

#     return relationship


# def get_mask_from_token_ids(token_ids: torch.IntTensor,
#                             padding_value: int = -1) -> torch.BoolTensor:
#     """
#     ...
#     Parameters
#     ----------
#      token_ids : torch.IntTensor
#         A padded tensor of shape (batch_size, seq_len).
#     """

#     mask = token_ids != padding_value
#     assert mask.dim() == 2
#     return mask


# def get_mask_from_embeddings(
#         embeddings: torch.FloatTensor) -> torch.BoolTensor:
#     """
#     ...
#     Parameters
#     ----------
#      embeddings : torch.FloatTensor
#         A padded tensor of shape (batch_size, seq_len, num_layers, embed_dim),
#         with the represenatations of the tokens.
#     """
#     mask = torch.sum(embeddings, dim=(_layer_dim, _emb_dim,)) != 0
#     assert mask.dim() == 2
#     return mask


# def check_loss_config(
#         losses: Union[Loss, Dict[str, Loss]],
#         loss_weights: Optional[Dict[str, float]],) -> None:

#     if isinstance(losses, dict):
#         assert losses.keys() == loss_weights.keys(), ConfigurationError
#     elif isinstance(losses, Loss):
#         assert loss_weights is None, ConfigurationError
#     else:
#         raise ConfigurationError
