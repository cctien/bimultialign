import logging
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Union

from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides
import torch

logger = logging.getLogger(__name__)


class PairDatasetReader(DatasetReader):
    """
    This ``DatasetReader`` ...

    # Parameters
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 data_dir: Optional[Union[Path, str]] = None,
                 pair_names: Optional[List[str]] = None,
                 all_lang_names: Optional[Union[List[str], Dict[str, List[str]]]] = None,
                 lazy: bool = False,
                 lazy_shuffle: bool = False,
                 cache_directory: Optional[str] = None,
                 **kwargs,) -> None:
        super().__init__(lazy=lazy, cache_directory=cache_directory, **kwargs)

        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self.data_dir = data_dir
        self.pair_names = pair_names
        self.all_lang_names = all_lang_names
        self.lazy_shuffle = lazy_shuffle

    @overrides
    def text_to_instance(self,
                         sentence_0: str,
                         sentence_1: str,
                         label: Optional[str] = None,) -> Instance:

        fields: Dict[str, Field] = {}
        # shape==(seq_len,)
        sentence_0 = self._tokenizer.tokenize(sentence_0)
        fields['sentence_0'] = TextField(sentence_0, self._token_indexers)
        sentence_1 = self._tokenizer.tokenize(sentence_1)
        fields['sentence_1'] = TextField(sentence_1, self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label)

        return Instance(fields)
