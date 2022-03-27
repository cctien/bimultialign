import logging
from typing import Dict, List, Optional


from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.models import Model
from allennlp.training.metrics import Metric
from overrides import overrides
import torch

from interlens.criterions.criterion import Criterion
from interlens.models.cycle_aligner import CycleAlignerModel
from interlens.modules.feedforward import FeedForward
from interlens.modules.lenses.lens import Lens
from interlens.modules.vector_similarities.vector_similarity import VectorSimilarity

logger = logging.getLogger('__name__')


@Model.register('gan_aligner')
class GanAlignerModel(CycleAlignerModel):
    """
    This ``Model`` ...

    """

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 lens: Lens,
                 vector_similarity: Optional[VectorSimilarity] = None,
                 criterion: Optional[Criterion] = None,
                 cycle: Optional[List[FeedForward]] = None,
                 critic: Optional[FeedForward] = None,
                 critic_criterion: Optional[Criterion] = None,
                 aggregate_criterion: Optional[Criterion] = None,
                 metrics: Optional[Dict[str, Metric]] = None,
                 initializer: Optional[InitializerApplicator] = None,
                 regularizer: Optional[RegularizerApplicator] = None,
                 **kwargs,) -> None:
        super().__init__(vocab=vocab,
                         text_field_embedder=text_field_embedder,
                         lens=lens,
                         vector_similarity=vector_similarity,
                         criterion=criterion,
                         cycle=cycle,
                         metrics=metrics,
                         regularizer=regularizer,
                         **kwargs)

        self.critic = critic
        self.critic_criterion = critic_criterion
        self.aggregate_criterion = aggregate_criterion

        for m in (self.critic,):
            for n, p in m.named_parameters():
                setattr(p, '_critic', True)
                logger.info(f"Set paramater {n} with _critic attr")

        for m in (self.lens, self.cycle,):
            for n, p in m.named_parameters():
                setattr(p, '_generator', True)
                logger.info(f"Set paramater {n} with _generator attr")

        if initializer is not None:
            logger.info(f"Initialize model parameters with {initializer}.")
            initializer(self)

    @overrides
    def forward(self,
                sentence_0: TextFieldTensors,
                sentence_1: TextFieldTensors,
                player: Optional[str],
                ) -> Dict[str, torch.Tensor]:

        if player == 'critic':
            with torch.no_grad():
                sentence_0 = self._get_sentence_encodings(sentence_0)
                sentence_1 = self._get_sentence_encodings(sentence_1)

            critic_loss = self._get_critic_loss(sentence_0.detach(),
                                                sentence_1.detach())

            # self.metrics.update({'critic_loss': critic_loss.detach().item(), })

            output_dict = {'loss': critic_loss, }

            return output_dict

        if player == 'generator' or player is None:
            sentence_0 = self._get_sentence_encodings(sentence_0)
            sentence_1 = self._get_sentence_encodings(sentence_1)

            critic_loss = self._get_critic_loss(sentence_0,
                                                sentence_1)
            cycle_loss = self._get_cycle_loss(sentence_0,
                                              sentence_1)
            generator_loss = self.aggregate_criterion(-critic_loss, cycle_loss)

            self.metrics.update({'critic_loss': critic_loss.detach().item(),
                                 'cycle_loss': cycle_loss.detach().item(),
                                 'generator_loss': generator_loss.detach().item(), })

            output_dict = {'loss': generator_loss, }

            return output_dict

        raise ValueError(f"Not recognized player {player}.")

    def _get_critic_loss(self,
                         sentence_0: torch.FloatTensor,
                         sentence_1: torch.FloatTensor,
                         ) -> torch.FloatTensor:

        return self.critic_criterion(self.critic(sentence_0),
                                     self.critic(sentence_1))

    @overrides
    def get_metrics(self, reset: bool = False,) -> Dict[str, float]:
        return self.metrics.copy()
