import logging
from typing import Optional

from allennlp.nn.activations import Activation
from overrides import overrides
import torch

from interlens.modules.layer_poolers.layer_pooler import LayerPooler

logger = logging.getLogger(__name__)


@LayerPooler.register('linear')
class LinearLayerPooler(LayerPooler):
    """
    `LinearLayerPooler` ...
    """

    def __init__(self,
                 num_layers: int = 25,
                 input_dim: int = 1024,
                 activation: Optional[Activation] = None,
                 trainable_scalar: bool = False,
                 dropout_p: Optional[float] = None,
                 do_layer_norm: bool = False,) -> None:
        super().__init__()

        if do_layer_norm:
            logger.info("Perform layer normalization with trainable parameters"
                        "upon the output embeddings.")
            trainable_scalar = False

        self.weight = torch.nn.Parameter(torch.ones(num_layers))
        self.activation = activation

        if trainable_scalar:
            self.scalar = torch.nn.Parameter(torch.ones(1))
        else:
            self.scalar = None

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

        weight = self.weight
        if self.activation is not None:
            weight = self.activation(weight)

        embeddings.transpose_(self._layer_dim, -1)
        embeddings = torch.matmul(embeddings, weight)

        if self.scalar is not None:
            embeddings = embeddings * self.scalar

        if self.dropout is not None:
            embeddings = self.dropout(embeddings)

        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)

        return embeddings
