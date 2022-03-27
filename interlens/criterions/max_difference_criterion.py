from typing import Optional

from allennlp.nn import Activation
from overrides import overrides
import torch

from interlens.criterions.criterion import Criterion


@Criterion.register('max_difference')
class MaxDifferenceCriterion(Criterion):
    def __init__(self,
                 activation: Optional[Activation] = None,
                 reduction: str = 'mean',
                 **kwargs,) -> None:
        super().__init__(reduction=reduction)

        self._activation = activation or (lambda x: x)

    @overrides
    def forward(self,
                scores_0: torch.FloatTensor,
                scores_1: torch.FloatTensor,
                ) -> torch.FloatTensor:

        return self._average(self._activation(scores_0)
                             - self._activation(scores_1))
