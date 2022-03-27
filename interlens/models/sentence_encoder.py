import logging
from typing import Dict, List, Optional, Tuple

from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Metric
from overrides import overrides
import torch

from interlens.criterions.criterion import Criterion
from interlens.modules.feedforward import FeedForward
from interlens.modules.lenses.lens import Lens
from interlens.modules.vector_similarities.vector_similarity import VectorSimilarity
from interlens.nn.util import get_two_slices

logger = logging.getLogger('__name__')
# torch.autograd.set_detect_anomaly(True)


@Model.register('sentence_encoder')
class SentenceEncoderModel(Model):
    """
    This ``Model`` ...

    # Parameters

    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    lens : ``Lens``, required.
    feedforward : ``FeedForward``
        ...
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 lens: Lens,
                 vector_similarity: Optional[VectorSimilarity] = None,
                 feedforward: Optional[FeedForward] = None,
                 criterion: Optional[Criterion] = None,
                 metrics: Optional[Dict[str, Metric]] = None,
                 initializer: Optional[InitializerApplicator] = None,
                 regularizer: Optional[RegularizerApplicator] = None,
                 **kwargs,) -> None:
        super().__init__(vocab=vocab, regularizer=regularizer, **kwargs)

        self.embedder = text_field_embedder
        self.lens = lens
        self.vector_similarity = vector_similarity
        self.feedforward = feedforward
        self.criterion = criterion
        self.metrics = metrics or {}

        if initializer is not None:
            logger.info(f"Initialize model parameters with {initializer}.")
            initializer(self)

    @overrides
    def forward(self,
                sentence: TextFieldTensors,
                ) -> Dict[str, torch.FloatTensor]:
        """Exists only for being able to save the baseline model for transfer easily.
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.feedforward is not None:
            loss = self.feedforward(torch.zeros(1024, device=device))
        else:
            loss = torch.zeros(1, device=device)

        output_dict = {'loss': loss}
        return output_dict

    def _get_sentence_encodings(self,
                                sentence: TextFieldTensors,
                                ) -> Dict[str, torch.FloatTensor]:

        embedded_sentence = self.embedder(sentence)
        encodings = self.lens(embedded_sentence,
                              mask=get_text_field_mask(sentence))
        return encodings

    def get_sentence_embeddings(self,
                                sentence: TextFieldTensors,
                                ) -> Dict[str, torch.FloatTensor]:

        return self._get_sentence_encodings(sentence)

    def get_sentence_similarity(self,
                                sentence_0: TextFieldTensors,
                                sentence_1: TextFieldTensors,
                                ) -> torch.FloatTensor:

        encoded_sentence_0 = self._get_sentence_encodings(sentence_0)
        encoded_sentence_1 = self._get_sentence_encodings(sentence_1)
        scores = self.vector_similarity(encoded_sentence_0, encoded_sentence_1)

        return scores

    def _get_combined_expressions(self,
                                  vec_0: torch.FloatTensor,
                                  vec_1: torch.FloatTensor,
                                  ) -> torch.FloatTensor:
        """
        # Input

        vec_0 : ``torch.FloatTensor``, required
            Shape: (batch_size, model_dim,)
        vec_1 : ``torch.FloatTensor``, required
            Shape: (batch_size, model_dim,)

        # Output

        scores : ``torch.FloatTensor``
            Shape: (batch_size, 4 * model_dim,)
        """

        difference = vec_0 - vec_1
        hadamard_product = vec_0 * vec_1
        combined_expression = torch.cat(
            (vec_0, vec_1, difference, hadamard_product,),
            dim=-1)

        return combined_expression

    def _update_metrics(self,
                        scores: torch.Tensor,
                        label: torch.Tensor,) -> None:

        for metric in self.metrics.values():
            metric(scores, label)

    @overrides
    def get_metrics(self, reset: bool = False,) -> Dict[str, float]:

        metrics = {metric_name: metric.get_metric(reset)
                   for metric_name, metric in self.metrics.items()}
        return metrics

    @overrides
    def make_output_human_readable(self,
                                   output_dict: Dict[str, torch.Tensor],
                                   ) -> Dict[str, torch.Tensor]:

        if 'logits' in output_dict and output_dict['logits'].dim() == 2:
            output_dict['probs'] = torch.nn.functional.softmax(
                output_dict['logits'], dim=-1)

        return output_dict

    # def get_output_dim(self,) -> int:
    #     return
