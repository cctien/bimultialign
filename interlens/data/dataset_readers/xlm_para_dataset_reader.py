import logging
from pathlib import Path
import random
from typing import Dict, Iterator, Tuple

from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Instance
from overrides import overrides

from interlens.data.dataset_readers.pair_dataset_reader import PairDatasetReader

logger = logging.getLogger(__name__)


@DatasetReader.register('xlm_para')
class XlmParaDatasetReader(PairDatasetReader):
    """
    This ``DatasetReader`` ...

    # Parameters
    """

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:

        file_paths = self._get_pair_file_paths(file_path)
        with open(cached_path(file_paths[0]), 'r') as data_file_0, \
                open(cached_path(file_paths[1]), 'r') as data_file_1:
            logger.info(f"Reading instances from file at: "
                        f"`{file_paths[0]}` and `{file_paths[1]}`.")

            if self.lazy_shuffle:
                para_texts = list(zip(data_file_0, data_file_1))
                random.shuffle(para_texts)
            else:
                para_texts = zip(data_file_0, data_file_1)

            para_texts = iter(para_texts)
            for line_0, line_1 in para_texts:

                line_0 = line_0.strip('\n')
                line_1 = line_1.strip('\n')

                if not line_0 or not line_1:
                    logger.warning(f"Encounter empty sentence.\n"
                                   f"sentence_0: {line_0},\nsentence_1: {line_1},\n"
                                   f"while reading instances from file at "
                                   f"{file_paths[0]} and {file_paths[1]}.")
                    continue

                yield self.text_to_instance(line_0, line_1)

    def _get_pair_file_paths(self, path: str) -> Tuple[Path]:

        path = Path(path)
        paired_str = '-'.join(sorted(self.pair_names))
        file_paths = (path.parent.joinpath(f"{paired_str}.{self.pair_names[0]}.{path.name}"),
                      path.parent.joinpath(f"{paired_str}.{self.pair_names[1]}.{path.name}"), )

        return file_paths
