from typing import Dict, Optional

from overrides import overrides
import torch

from interlens.criterions.criterion import Criterion


@Criterion.register('cross_entropy')
class CrossEntropyCriterion(Criterion):
    def __init__(self,
                 reduction: str = 'mean',
                 **kwargs,) -> None:
        super().__init__()

        self._cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                        **kwargs)

    @overrides
    def forward(self,
                logits: torch.Tensor,
                target: torch.Tensor,
                ) -> torch.Tensor:

        return self._cross_entropy(logits, target)
