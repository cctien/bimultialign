from typing import Dict

from overrides import overrides
import torch

from interlens.modules.lenses.lens import Lens
from interlens.nn.util import get_unit_normalized_vector


@Lens.register('basic')
class BasicLens(Lens):

    @overrides
    def forward(self,
                embeddings: torch.FloatTensor,
                mask: torch.BoolTensor,
                ) -> Dict[str, torch.FloatTensor]:

        # shape==(batch_size, seq_len, embed_dim)
        embeddings = self.layer_pooler(embeddings, mask)
        embeddings = self.seq_encoder(embeddings, mask)
        # shape==(batch_size, embed_dim)
        embeddings = self.vec_encoder(embeddings, mask)

        if self.feedforward is not None:
            embeddings = self.feedforward(embeddings)

        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)

        # slices = self._get_slices(embeddings)

        if self.norm_unit_slice:
            embeddings = get_unit_normalized_vector(embeddings)

        return embeddings
