from overrides import overrides
import torch

from interlens.modules.layer_poolers.layer_pooler import LayerPooler


@LayerPooler.register('sum')
class SumLayerPooler(LayerPooler):
    """
    `SumLayerPooler` ...
    """

    def __init__(self,
                 averaged: bool = True,) -> None:
        super().__init__()

        if averaged:
            self._average = torch.mean
        else:
            self._average = torch.sum

    @overrides
    def forward(self,
                embeddings: torch.FloatTensor,
                mask: torch.BoolTensor,) -> torch.FloatTensor:

        return self._average(embeddings, self._layer_dim)
