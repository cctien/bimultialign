import logging
from typing import Dict, Optional

from allennlp.nn.util import min_value_of_dtype
from overrides import overrides
import torch

from interlens.criterions.criterion import Criterion
from interlens.criterions.contrastive.contrastive_criterion import ContrastiveCriterion
from interlens.nn.util import get_argmax_ind, get_diag_mask

logger = logging.getLogger(__name__)


@Criterion.register('triplet_ranking')
class TripletRankingCriterion(ContrastiveCriterion):
    """
    Computes the Triplet ranking loss with mining for negatives.
    The implementation follows that of [VSE++](https://github.com/fartashf/vsepp/blob/master/model.py#L245).

    Registered as a `Criterion` with name 'triplet_ranking'.
    """

    @overrides
    def forward(self,
                scores: torch.FloatTensor,
                ) -> torch.FloatTensor:
        """
        # Input

            scores: `torch.FloatTensor`
                Shape: `(batch_size, batch_size)`
            output_dict `Dict[str, torch.Tensor]`
        """

        batch_size = scores.shape[0]
        diag_mask = get_diag_mask(batch_size, 2, device=scores.device)

        scores_positive = torch.diagonal(scores, offset=0)
        scores_positive_stack = torch.stack((
            # Correct row extraction for every column.
            scores_positive.view(1, -1).expand(batch_size, -1),
            # Correct column extraction for every row.
            scores_positive.view(-1, 1).expand(-1, batch_size),),
            dim=0)
        costs = scores.expand(2, -1, -1) - scores_positive_stack

        costs.masked_fill_(diag_mask, min_value_of_dtype(costs.dtype))

        # Draw for negative samples randomly.
        mask = self._get_sampling_mask(costs, batch_size)
        # Impose the hinge by masking-out easy triplets.
        if self.margin is not None:
            mask.masked_fill_(costs + self.margin < 0, False)

        # Mine for the hardest negatives.
        if self.mine_hardest_negatives:
            for dim in self._compared_dims:
                mask[dim][get_argmax_ind(costs[dim], dim)] = True

        # Clear the differences between positive samples themselves.
        mask.masked_fill_(diag_mask, False)

        costs = torch.masked_select(costs, mask)
        # logger.debug(f"mask:\n{mask}")
        # logger.debug(f"costs:\n{costs}")

        return self._average(self._activation(costs))
