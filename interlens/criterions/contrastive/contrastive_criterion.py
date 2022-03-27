import logging
from typing import Dict, Optional

from allennlp.common.checks import ConfigurationError
from allennlp.nn import Activation
import torch
from overrides import overrides

from interlens.criterions.criterion import Criterion
from interlens.nn.util import get_diag_mask

logger = logging.getLogger(__name__)


class ContrastiveCriterion(Criterion):
    def __init__(self,
                 num_random_samples: Optional[int] = None,
                 margin: Optional[float] = None,
                 mine_hardest_negatives: bool = True,
                 activation: Optional[Activation] = None,
                 reduction: str = 'mean',) -> None:
        super().__init__(reduction=reduction)

        if num_random_samples is None:
            if margin is None:
                mine_hardest_negatives = False
                logger.info("Use all samples in the mini-batch.")
            else:
                logger.info("Use all samples filterd by the hinge.")
        elif num_random_samples > 0:
            logger.info(f"Draw {num_random_samples} negative samples "
                        "for every input on average.")
        elif num_random_samples == 0:
            logger.info("Not draw random samples.")
        else:
            raise ConfigurationError("`num_random_samples` must be nonnegative number"
                                     f" or `None`. Got {num_random_samples} instead.")

        if mine_hardest_negatives:
            logger.info("Mine for and include the hardest negatives.")

        self.num_random_samples = num_random_samples
        self.margin = margin
        self.mine_hardest_negatives = mine_hardest_negatives
        self._activation = activation or (lambda x: x)

        self._compared_dims = (0, 1,)

    def _get_sampling_mask(self,
                           inputs: torch.Tensor,
                           batch_size: int,) -> torch.BoolTensor:

        if self.num_random_samples is None:
            sampling_mask = torch.ones_like(inputs, dtype=torch.bool)
        elif self.num_random_samples == 0:
            sampling_mask = torch.zeros_like(inputs, dtype=torch.bool)
        else:
            neg_samp_prob = torch.true_divide(self.num_random_samples,
                                              (batch_size - 1))
            sampling_mask = torch.rand_like(inputs) < neg_samp_prob
            # logger.debug(f"neg_sampling_prob: {neg_samp_prob}")

        # logger.debug(f"Sampling_mask: {sampling_mask}.")

        return sampling_mask

    # @overrides
    # def forward(self,
    #             scores: torch.FloatTensor,
    #             ) -> torch.FloatTensor:
    #     raise NotImplementedError

    # @overrides
    # def _forward_verbose(self,
    #                      scores: torch.Tensor,
    #                      output_dict: Dict[str, torch.Tensor],) -> None:

    #     positive_mask = get_diag_mask(scores.shape[0], device=scores.device)
    #     scores_pos = torch.masked_select(scores, mask=positive_mask)
    #     scores_neg = torch.masked_select(scores, mask=~positive_mask)
    #     output_dict['scores_positives'] = torch.mean(scores_pos)
    #     output_dict['scores_negatives'] = torch.mean(scores_neg)
