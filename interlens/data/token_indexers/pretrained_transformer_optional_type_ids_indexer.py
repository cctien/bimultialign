from typing import List

from allennlp.data import IndexedTokenList, Token, TokenIndexer, Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from overrides import overrides


@TokenIndexer.register('pretrained_transformer_optional_type_ids')
class PretrainedTransformerOptionalTypeIdsIndexer(PretrainedTransformerIndexer):
    """
    This `TokenIndexer`...
    # Parameters

    model_name : `str`
        The name of the `transformers` model to use.
    namespace : `str`, optional (default=`tags`)
        We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
        We use a somewhat confusing default value of `tags` so that we do not add padding or UNK
        tokens to this namespace, which would break on loading because we wouldn't find our default
        OOV token.
    max_length : `int`, optional (default = `None`)
        If not None, split the document into segments of this many tokens (including special tokens)
        before feeding into the embedder. The embedder embeds these segments independently and
        concatenate the results to get the original document representation. Should be set to
        the same value as the `max_length` option on the `PretrainedTransformerEmbedder`.
    add_type_ids: `bool`, optional (default = `False`)
        If True, the `IndexedTokenList` returned by `TokenIndexer.tokens_to_indices`
        and `DataArray` returned by `Batch.as_tensor_dict` have keys `type_ids`; otherwise not so.
    """

    def __init__(self,
                 model_name: str = 'xlm-roberta-large',
                 namespace: str = 'tags',
                 max_length: int = None,
                 add_type_ids: bool = False,
                 **kwargs,) -> None:
        super().__init__(model_name=model_name,
                         namespace=namespace,
                         max_length=max_length,
                         **kwargs)

        self.add_type_ids = add_type_ids

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,) -> IndexedTokenList:
        self._add_encoding_to_vocabulary_if_needed(vocabulary)

        indices, type_ids = self._extract_token_and_type_ids(tokens)
        output: IndexedTokenList = {'token_ids': indices,
                                    'mask': [True] * len(indices), }
        if self.add_type_ids:
            output['type_ids'] = type_ids

        return self._postprocess_output(output)
