from overrides import overrides
import torch

from interlens.modules.layer_poolers.layer_pooler import LayerPooler


@LayerPooler.register('one')
class OneLayerPooler(LayerPooler):
    """
    `OneLayerPooler` ...
    """

    def __init__(self, layer: int,) -> None:
        super().__init__()
        self.layer_ind = layer

    @overrides
    def forward(self,
                embeddings: torch.FloatTensor,
                mask: torch.BoolTensor,) -> torch.FloatTensor:

        return torch.squeeze(torch.narrow(embeddings, self._layer_dim, self.layer_ind, 1),
                             self._layer_dim)
