from allennlp.nn.util import tiny_value_of_dtype
import torch
from overrides import overrides

from interlens.modules.vector_similarities.vector_similarity import VectorSimilarity


@VectorSimilarity.register('dot_product')
class DotProductVectorSimilarity(VectorSimilarity):
    """
    This `VectorSimilarity` is a `Module` that ...
    """

    @overrides
    def forward(self,
                vec_0: torch.FloatTensor,
                vec_1: torch.FloatTensor,) -> torch.FloatTensor:

        return torch.mm(vec_0, vec_1.transpose(-2, -1))
