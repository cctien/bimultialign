import math
from typing import Optional

from overrides import overrides
import torch

from interlens.modules.layer_poolers.layer_pooler import LayerPooler


@LayerPooler.register('softmax')
@LayerPooler.register('softmax_linear')
class SoftmaxLinearLayerPooler(LayerPooler):
    """
    `SoftmaxLinearLayerPooler` ...
    """

    def __init__(self,
                 num_layers: int = 25,
                 input_dim: int = 1024,
                 dropout_p: Optional[float] = None,
                 do_layer_norm: bool = False,) -> None:
        super().__init__()

        self.weight = torch.nn.Parameter(torch.zeros(num_layers))
        self.scalar = torch.nn.Parameter(torch.ones(1))

        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=dropout_p)
        else:
            self.dropout = None

        if do_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(input_dim)
        else:
            self.layer_norm = None

    @overrides
    def forward(self,
                embeddings: torch.FloatTensor,
                mask: torch.BoolTensor,) -> torch.FloatTensor:

        normed_weights = torch.nn.functional.softmax(self.weight, dim=-1)

        embeddings = torch.transpose(embeddings, self._layer_dim, -1)
        embeddings = self.scalar * torch.matmul(embeddings, normed_weights)

        if self.dropout is not None:
            embeddings = self.dropout(embeddings)

        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)

        return embeddings
