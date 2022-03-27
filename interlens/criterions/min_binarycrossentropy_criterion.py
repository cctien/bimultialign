from overrides import overrides
import torch

from interlens.criterions.criterion import Criterion


@Criterion.register('min_binarycrossentropy')
class MinBinarycrossentropyCriterion(Criterion):
    def __init__(self,
                 reduction: str = 'mean',
                 **kwargs,) -> None:
        super().__init__(reduction=reduction)

        self._bcewithlogits = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    @overrides
    def forward(self,
                scores_0: torch.FloatTensor,
                scores_1: torch.FloatTensor,
                ) -> torch.FloatTensor:

        return self._bcewithlogits(torch.cat([scores_0,
                                              scores_1, ]),
                                   torch.cat([torch.zeros_like(scores_0),
                                              torch.ones_like(scores_1), ]))
