import logging
from typing import Optional

from allennlp.modules.token_embedders import TokenEmbedder, PretrainedTransformerEmbedder
from allennlp.nn.util import batched_index_select
from overrides import overrides
import torch
from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)
_layers_dim: int = -2
_embeddings_dim: int = -1


@TokenEmbedder.register('pretrained_transformer_all_layers')
class PretrainedTransformerAllLayersEmbedder(PretrainedTransformerEmbedder):
    """
    Uses a pretrained model from `transformers` as a `TokenEmbedder`.
    Registered as a `TokenEmbedder` with name "pretrained_transformer_all_layers".

    This module inherits and is only slightly modified from [Allennlp](https://github.com/allenai/allennlp/blob/master/allennlp/modules/token_embedders/pretrained_transformer_embedder.py).

    The key difference is that the module returns all hidden states, 
    and it is up to the later layer to execute the layer pooling such as scalar mixure, etc.
    Another difference, which is a hack, is that the pretrained embedder is not registered with 
    parameters by Pytorch. This is done only to save storage when saving the models.

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerIndexer`.
    max_length : `int`, optional (default = `None`)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedTransformerIndexer`.
    sub_module: `str`, optional (default = `None`)
        The name of a submodule of the transformer to be used as the embedder. Some transformers naturally act
        as embedders such as BERT. However, other models consist of encoder and decoder, in which case we just
        want to use the encoder.
    """

    def __init__(self,
                 model_name: str = 'xlm-roberta-large',
                 max_length: Optional[int] = None,
                 sub_module: Optional[str] = None,
                 train_parameters: bool = False,
                 override_weights_file: Optional[str] = None,
                 override_weights_strip_prefix: Optional[str] = None,
                 ) -> None:
        super().__init__(model_name=model_name,
                         max_length=max_length,
                         sub_module=sub_module,
                         train_parameters=train_parameters,
                         last_layer_only=False,
                         override_weights_file=override_weights_file,
                         override_weights_strip_prefix=override_weights_strip_prefix)

        transformer_model = self.transformer_model
        del self.transformer_model, self._scalar_mix

        if torch.cuda.is_available():
            transformer_model.cuda()
        else:
            logger.warning("Pretrained transformer embedder not on GPU.")

        transformer_model.eval()
        self.transformer_model = transformer_model.__call__

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: `torch.LongTensor`
            Shape: `[batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces]`.
            num_segment_concat_wordpieces is num_wordpieces plus special tokens inserted in the
            middle, e.g. the length of: "[CLS] A B C [SEP] [CLS] D E F [SEP]" (see indexer logic).
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Shape: `[batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces]`.
        segment_concat_mask: `Optional[torch.BoolTensor]`
            Shape: `[batch_size, num_segment_concat_wordpieces]`.

        # Returns

        `torch.Tensor`
            Shape: `[batch_size, num_wordpieces, num_layers, embedding_size]`.

        """
        # This module does not make use of type ids.
        type_ids = None

        # # Some of the huggingface transformers don't support type ids at all and crash when you supply
        # # them. For others, you can supply a tensor of zeros, and if you don't, they act as if you did.
        # # There is no practical difference to the caller, so here we pretend that one case is the same
        # # as another case.
        # if type_ids is not None:
        #     max_type_id = type_ids.max()
        #     if max_type_id == 0:
        #         type_ids = None
        #     else:
        #         if max_type_id >= self._number_of_token_type_embeddings():
        #             raise ValueError("Found type ids too large "
        #                              "for the chosen transformer model.")
        #         assert token_ids.shape == type_ids.shape

        fold_long_sequences = (self._max_length is not None
                               and token_ids.size(1) > self._max_length)
        if fold_long_sequences:
            batch_size, num_segment_concat_wordpieces = token_ids.size()
            token_ids, segment_concat_mask, type_ids = self._fold_long_sequences(
                token_ids, segment_concat_mask, type_ids)

        transformer_mask = segment_concat_mask if self._max_length is not None else mask
        # Shape: [batch_size, num_wordpieces, embedding_size],
        # or if self._max_length is not None:
        # [batch_size * num_segments, self._max_length, embedding_size]

        # We call this with kwargs because some of the huggingface models don't have the
        # token_type_ids parameter and fail even when it's given as None.
        # Also, as of transformers v2.5.1, they are taking FloatTensor masks.
        parameters = {"input_ids": token_ids,
                      "attention_mask": transformer_mask.float()}
        if type_ids is not None:
            parameters["token_type_ids"] = type_ids

        transformer_output = self.transformer_model(**parameters)
        hidden_states = transformer_output[-1]
        embeddings = torch.stack(hidden_states, dim=-2).detach()

        if fold_long_sequences:

            embeddings = embeddings.reshape(
                *segment_concat_mask.shape, len(hidden_states) * self.get_output_dim())
            embeddings = self._unfold_long_sequences(
                embeddings, segment_concat_mask, batch_size, num_segment_concat_wordpieces)
            embeddings = embeddings.reshape(
                *mask.shape, len(hidden_states), self.get_output_dim())

        return embeddings
