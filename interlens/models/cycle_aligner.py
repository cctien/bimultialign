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
from interlens.models.sentence_encoder import SentenceEncoderModel
from interlens.modules.feedforward import FeedForward
from interlens.modules.lenses.lens import Lens
from interlens.modules.vector_similarities.vector_similarity import VectorSimilarity
from interlens.nn.util import get_unit_normalized_vector

logger = logging.getLogger('__name__')


@Model.register('cycle_aligner')
class CycleAlignerModel(SentenceEncoderModel):
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
                 metrics: Optional[Dict[str, Metric]] = None,
                 initializer: Optional[InitializerApplicator] = None,
                 regularizer: Optional[RegularizerApplicator] = None,
                 **kwargs,) -> None:
        super().__init__(vocab=vocab,
                         text_field_embedder=text_field_embedder,
                         lens=lens,
                         vector_similarity=vector_similarity,
                         criterion=criterion,
                         metrics=metrics,
                         regularizer=regularizer,
                         **kwargs)

        self.cycle = torch.nn.ModuleList(cycle)

        if initializer is not None:
            logger.info(f"Initialize model parameters with {initializer}.")
            initializer(self)

    @overrides
    def forward(self,
                sentence_0: TextFieldTensors,
                sentence_1: TextFieldTensors,
                ) -> Dict[str, torch.Tensor]:

        sentence_0 = self._get_sentence_encodings(sentence_0)
        sentence_1 = self._get_sentence_encodings(sentence_1)

        cycle_loss = self._get_cycle_loss(sentence_0, sentence_1)
        output_dict = {'loss': cycle_loss, }

        return output_dict

    def _get_cycle_loss(self,
                        sentence_0: torch.FloatTensor,
                        sentence_1: torch.FloatTensor,
                        ) -> torch.FloatTensor:

        if not self.cycle:
            return torch.as_tensor(0, device=sentence_0.device)

        cycled_sentence_0 = self.cycle[1](self.cycle[0](sentence_0))
        cycled_sentence_1 = self.cycle[0](self.cycle[1](sentence_1))

        if self.lens.norm_unit_slice:
            cycled_sentence_0 = get_unit_normalized_vector(cycled_sentence_0)
            cycled_sentence_1 = get_unit_normalized_vector(cycled_sentence_1)

        return torch.add(
            self.criterion(self.vector_similarity(sentence_0,
                                                  cycled_sentence_0)),
            self.criterion(self.vector_similarity(sentence_1,
                                                  cycled_sentence_1)))
