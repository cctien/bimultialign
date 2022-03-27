import math

from allennlp.common import Registrable
import torch

from overrides import overrides


class LayerPooler(torch.nn.Module, Registrable):
    """
    A `LayerPooler` is a `Module` which aggregates information from all layers.
    """

    _layer_dim: int = -2

    @overrides
    def forward(self,
                embeddings: torch.FloatTensor,
                mask: torch.BoolTensor,) -> torch.FloatTensor:
        """
        # Input

        embeddings : `torch.FloatTensor`
            shape==(batch_size, seq_len, num_layers, model_dim,)
        mask : `torch.BoolTensor`
            shape==(batch_size, seq_len,)

        # Output

        embeddings : `torch.FloatTensor`
            shape==(batch_size, seq_len, model_dim,)
        """

        raise NotImplementedError

    # def _reset_weight_parameter(self, num_layers: int,) -> None:
    #     bound = 1.0 / math.sqrt(num_layers)
    #     torch.nn.init.uniform_(self.weight, -bound, bound)
