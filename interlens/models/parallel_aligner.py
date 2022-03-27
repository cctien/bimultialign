from typing import Dict, List

from allennlp.data import TextFieldTensors
from allennlp.models import Model
from overrides import overrides
import torch

from interlens.models.sentence_encoder import SentenceEncoderModel


@Model.register('parallel_aligner')
class ParallelAlignerModel(SentenceEncoderModel):
    """
    This ``Model`` ...

    """

    @overrides
    def forward(self,
                sentence_0: TextFieldTensors,
                sentence_1: TextFieldTensors,
                ) -> Dict[str, torch.Tensor]:

        encoded_sentence_0 = self._get_sentence_encodings(sentence_0)
        encoded_sentence_1 = self._get_sentence_encodings(sentence_1)
        scores = self.vector_similarity(encoded_sentence_0, encoded_sentence_1)
        output_dict = {'loss': self.criterion(scores), }

        return output_dict
