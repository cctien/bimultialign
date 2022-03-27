from typing import Dict, Optional

from allennlp.common import Registrable
from allennlp.modules import LayerNorm, Seq2SeqEncoder, Seq2VecEncoder
from overrides import overrides
import torch

from interlens.modules.feedforward import FeedForward
from interlens.modules.layer_poolers.layer_pooler import LayerPooler
from interlens.nn.util import get_two_slices, get_unit_normalized_vector


class Lens(torch.nn.Module, Registrable):
    """
    A `Lens` is a `Module` that ...


    # Parameters

    layer_pooler : ``LayerPooler``
        ...
    sentence_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the abstract to a vector.
    inferencer : ``FeedForward``
        ...
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.

    """

    default_implementation = 'basic'

    def __init__(self,
                 layer_pooler: LayerPooler,
                 seq_encoder: Seq2SeqEncoder,
                 vec_encoder: Seq2VecEncoder,
                 feedforward: Optional[FeedForward] = None,
                 layer_norm: Optional[LayerNorm] = None,
                 intra_slice_dim: int = 0,
                 norm_unit_slice: bool = False,
                 ) -> None:
        super().__init__()

        self.layer_pooler = layer_pooler
        self.seq_encoder = seq_encoder
        self.vec_encoder = vec_encoder
        self.feedforward = feedforward
        self.layer_norm = layer_norm
        self.intra_slice_dim = intra_slice_dim
        self.norm_unit_slice = norm_unit_slice

        if self.feedforward is not None:
            self._output_dim = self.feedforward.get_output_dim()
        else:
            self._output_dim = self.vec_encoder.get_output_dim()

    def get_inter_dim(self,) -> int:
        return self._output_dim - self.intra_slice_dim

    def get_intra_dim(self,) -> int:
        return self.intra_slice_dim

    def _get_slices(self,
                    embeddings: torch.FloatTensor,
                    ) -> Dict[str, torch.FloatTensor]:
        slices = {}
        slices['inter'], slices['intra'] = get_two_slices(embeddings,
                                                          self._output_dim,
                                                          self.intra_slice_dim)

        if self.norm_unit_slice:
            for k, v in slices.items():
                slices[k] = get_unit_normalized_vector(v)

        return slices

    # def get_output_dim(self) -> int:
    #     return self._output_dim
