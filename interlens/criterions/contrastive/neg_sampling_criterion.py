import logging
from typing import Dict, Optional

from overrides import overrides
import torch

from interlens.criterions.criterion import Criterion
from interlens.criterions.contrastive.contrastive_criterion import ContrastiveCriterion
from interlens.nn.util import get_argmin_ind, get_diag_mask

logger = logging.getLogger(__name__)


@Criterion.register('neg_sampling')
class NegSamplingCriterion(ContrastiveCriterion):
    """
    Computes the logistic loss with negative sampling as
    adapted from [Mikolov et al. 2013. Distributed representations of words](https://arxiv.org/abs/1310.4546).

    Registered as a `Criterion` with name 'neg_sampling'.
    """

    @overrides
    def forward(self,
                scores: torch.FloatTensor,
                ) -> torch.FloatTensor:
        """
        # Input

            scores: `torch.FloatTensor`
                Shape: `(batch_size, batch_size)`
        """

        batch_size = scores.shape[0]
        diag_mask = get_diag_mask(batch_size, device=scores.device)

        # Negate the scores of negative samples for binary cross entropy loss or JS divergence,
        # and hard negatives mining.
        scores_positive = torch.diagonal(scores)
        if self.margin is not None:
            scores_positive = scores_positive - self.margin

        scores = torch.neg(scores)
        scores.masked_scatter_(diag_mask, scores_positive)

        # Draw random negative samples in the mini-batch.
        mask = self._get_sampling_mask(scores, batch_size)

        # Mine for the hardest negatives.
        if self.mine_hardest_negatives:
            for dim in self._compared_dims:
                mask[get_argmin_ind(scores, dim)] = True

        # Use all positive samples.
        mask.masked_fill_(diag_mask, True)

        scores = torch.masked_select(scores, mask)
        # logger.debug(f"scores:\n{scores}")
        # logger.debug(f"mask:\n{mask}")

        return torch.neg(self._average(self._activation(scores)))
