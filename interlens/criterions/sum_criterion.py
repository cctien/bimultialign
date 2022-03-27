from typing import List, Union

from overrides import overrides
import torch

from interlens.criterions.criterion import Criterion


@Criterion.register('sum')
@Criterion.register('aggregate')
class SumCriterion(Criterion):
    """
    This `Criterion`, in the `forward` method, sum the input losses as weighted by `lambdas_weights`.
    The weight for the first element of `losses` is fixed to be `1.0`,
    and the number of losses given the `forward` must equal to `len(lambdas_weights) + 1`

    # Paramters

        lamda_weights: `List[float]`.
             Lambda weights or multiplier for every loss since the second one, in such order.
    """

    def __init__(self,
                 lamda_weights: List[Union[float, int]],
                 **kwargs,) -> None:
        super().__init__()

        lamda_weights = ([1.0, ]
                         + [float(num) for num in lamda_weights])
        _lamda_weights = torch.as_tensor(lamda_weights)
        if torch.cuda.is_available():
            _lamda_weights = _lamda_weights.cuda()

        self._lamda_weights = _lamda_weights

    @ overrides
    def forward(self,
                *losses,
                ) -> torch.FloatTensor:

        return torch.sum(torch.stack(losses) * self._lamda_weights)
