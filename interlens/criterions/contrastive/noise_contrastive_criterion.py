import logging
from typing import Dict, Optional

from overrides import overrides
import torch

from interlens.criterions.criterion import Criterion
from interlens.criterions.contrastive.contrastive_criterion import ContrastiveCriterion
from interlens.nn.util import get_argmax_ind, get_diag_mask

logger = logging.getLogger(__name__)


@Criterion.register('noise_contrastive')
class NoiseContrastiveCriterion(ContrastiveCriterion):
    """
    Computes the information noise contrastive estimation criterion or infoNCE
    according to [Oord et al. 2019. Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748).

    Registered as a `Criterion` with name 'noise_contrastive'.
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

        if self.margin is not None:
            scores = scores - diag_mask * self.margin

        # Draw random negative samples in the mini-batch.
        mask = self._get_sampling_mask(scores, batch_size)

        # Mine for the hardest negatives.
        if self.mine_hardest_negatives:
            for dim in self._compared_dims:
                mask[get_argmax_ind(scores, dim, ~diag_mask)] = True

        # Use all positive samples.
        mask.masked_fill_(diag_mask, True)

        scores.masked_fill_(~mask, float('-inf'))

        # Shape: (2, batch_size, batch_size)
        scores = torch.stack((torch.nn.functional.log_softmax(scores, 0),
                              torch.nn.functional.log_softmax(scores, 1),),
                             0)
        scores = torch.masked_select(scores, get_diag_mask(batch_size, 2,
                                                           device=scores.device))
        # logger.debug(f"mask:\n{mask}")
        # logger.debug(f"scores:\n{scores}")

        return torch.neg(self._average(scores))
