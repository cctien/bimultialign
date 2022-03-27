from typing import Union

from allennlp.nn.util import tiny_value_of_dtype
import torch
from overrides import overrides

from interlens.modules.vector_similarities.vector_similarity import VectorSimilarity


@VectorSimilarity.register('minus_p_distance')
class MinusDistanceVectorSimilarity(VectorSimilarity):
    """
    This `VectorSimilarity` is a `Module` that ...
    """

    def __init__(self,
                 p: Union[float, int] = 2,
                 use_raised_distance: bool = True,) -> None:
        super().__init__()

        self.p = p
        self.use_raised_distance = use_raised_distance

    @overrides
    def forward(self,
                vec_0: torch.FloatTensor,
                vec_1: torch.FloatTensor,) -> torch.FloatTensor:

        # dist = torch.cdist(vec1, vec2, p=self.p)
        # return torch.neg(dist)

        vec_0 = torch.unsqueeze(vec_0, 1)
        vec_1 = torch.unsqueeze(vec_1, 0)

        dist_raised = torch.sum(torch.pow(vec_0 - vec_1, self.p),
                                -1)

        if self.use_raised_distance:
            return torch.neg(dist_raised)
        else:
            return torch.neg(torch.pow(dist_raised,
                                       torch.true_divide(1, self.p)))
