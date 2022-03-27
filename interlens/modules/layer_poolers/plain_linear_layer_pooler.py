from typing import Optional

from overrides import overrides
import torch

from interlens.modules.layer_poolers.layer_pooler import LayerPooler


@LayerPooler.register('plain_linear')
class PlainLinearLayerPooler(LayerPooler):
    """
    `LinearLayerPooler` ...
    """

    def __init__(self,
                 num_layers: int = 25,
                 input_dim: int = 1024,
                 dropout_p: Optional[float] = None,
                 do_layer_norm: bool = False,) -> None:
        super().__init__()

        self.linear_layer = torch.nn.Linear(num_layers, 1, bias=True)

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

        embeddings = torch.transpose(embeddings, self._layer_dim, -1)
        embeddings = torch.squeeze(self.linear_layer(embeddings), dim=-1)
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)

        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)

        return embeddings
