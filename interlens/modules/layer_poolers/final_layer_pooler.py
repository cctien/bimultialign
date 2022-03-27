from overrides import overrides
import torch

from interlens.modules.layer_poolers.layer_pooler import LayerPooler


@LayerPooler.register('final')
class FinalLayerPooler(LayerPooler):
    """
    `FinalLayerPooler` ...
    """

    @overrides
    def forward(self,
                embeddings: torch.FloatTensor,
                mask: torch.BoolTensor,) -> torch.FloatTensor:

        return torch.squeeze(torch.narrow(embeddings, self._layer_dim, -1, 1),
                             self._layer_dim)
