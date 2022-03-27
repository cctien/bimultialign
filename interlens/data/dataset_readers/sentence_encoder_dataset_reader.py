import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
import numpy
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register('sentence_encoder')
class SentenceEncoderDatasetReader(DatasetReader):
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
                 number_of_instances: Optional[int] = None,
                 lazy: bool = False,
                 cache_directory: Optional[str] = None,
                 **kwargs,) -> None:
        super().__init__(lazy=lazy, cache_directory=cache_directory, **kwargs)

        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self.data_dir = data_dir
        self.pair_names = pair_names
        self.all_lang_names = all_lang_names
        self.number_of_instances = number_of_instances
        logger.info(f"pair_names {pair_names} unused.")

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(cached_path(file_path), 'r') as data_file:
            logger.info(f"Reading instances from file at: {file_path}.")

            for line in data_file:
                line = line.strip('\n')
                if not line:
                    continue

                yield self.text_to_instance(line)

    @overrides
    def text_to_instance(self,
                         sentence: str,
                         ) -> Instance:

        # shape==(seq_len,)
        sentence = self._tokenizer.tokenize(sentence)
        fields = {'sentence': TextField(sentence, self._token_indexers), }

        return Instance(fields)

    # def numpy_batch_to_batch_tensor_dict(self,
    #                                      batch: numpy.ndarray,
    #                                      ) -> TensorDict:
    #     """
    #     Read a batch of raw texts as a `numpy.ndarray`, from the dataset by `senteval`.
    #     """

    #     # instances = [self.text_to_instance(s.item()) for s in batch]

    #     instances = [self.text_to_instance(' '.join(sent)) for sent in batch]
    #     vocab = Vocabulary.from_instances(instances)
    #     batch = Batch(instances)
    #     batch.index_instances(vocab)
    #     return batch.as_tensor_dict(batch.get_padding_lengths())
