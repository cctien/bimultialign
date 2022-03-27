from allennlp.common import Registrable
import torch

from overrides import overrides


class VectorSimilarity(torch.nn.Module, Registrable):
    """
    A `VectorSimilarity` is a `Module` that ...
    """

    @overrides
    def forward(self,
                vec_0: torch.FloatTensor,
                vec_1: torch.FloatTensor,) -> torch.FloatTensor:
        """
        # Input

        vec_0 : ``torch.FloatTensor``, required
            Shape: (batch_size, model_dim,)
        vec_1 : ``torch.FloatTensor``, required
            Shape: (batch_size, model_dim,)

        # Output
        scores : ``torch.FloatTensor``
            Shape: (batch_size, batch_size,)
        """

        raise NotImplementedError
