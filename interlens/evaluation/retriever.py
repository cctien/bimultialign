import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from allennlp.common import Registrable
import numpy
import torch

from interlens.evaluation.util_retriever import bucc_eval, mine_bitext, generic_eval

logger = logging.getLogger(__name__)


class Retriever(Registrable):
    """
    A `Retriever` ...
    """

    default_implementation = 'basic'

    def __init__(self,
                 task_name: str,
                 use_shift_embeds: bool = False,
                 similarity_metric: str = 'cosine',
                 neighborhood_size: int = 4,
                 #  use_gpu: bool = False,
                 ) -> None:

        self.task_name = task_name
        self.use_shift_embeds = use_shift_embeds
        self.similarity_metric = similarity_metric
        self.neighborhood_size = neighborhood_size
        if similarity_metric not in ('cosine', ):
            raise NotImplementedError

        # if use_gpu:
        #     if torch.cuda.is_available():
        #         logger.info("Use GPU for KNN search with Faiss.")
        #         self.use_gpu = True
        #     else:
        #         logger.warning("GPU unavailable. Fallback to CPU.")
        #         self.use_gpu = False
        # else:
        #     logger.info("Use CPU for KNN search.")
        #     self.use_gpu = False


@Retriever.register('basic')
class BasicRetriever(Retriever):
    """
    `BasicRetriever` is a `Retriever` ...
    """

    def retrieve(self,
                 embeddings_src: numpy.ndarray,
                 embeddings_trg: numpy.ndarray,
                 src_text_file: Path,
                 trg_text_file: Path,
                 src_ind_file: Path,
                 trg_ind_file: Path,
                 cand2score_file: Path,
                 retrieval_scheme: str = 'max',
                 margin_scheme: str = 'ratio',
                 threshold: Optional[float] = None,
                 gold_file: Optional[Path] = None,
                 predict_file: Optional[Path] = None,
                 use_gpu: bool = False,
                 ) -> Dict[str, float]:

        if cand2score_file.is_file():
            logger.info("Use existing scores at {cand2score_file}.")
        else:
            logger.info("Compute the similarity scores. "
                        f"use_gpu: {use_gpu}.")
            mine_bitext(embeddings_src, embeddings_trg,
                        src_text_file, trg_text_file,
                        cand2score_file,
                        dist=self.similarity_metric,
                        neighborhood=self.neighborhood_size,
                        retrieval=retrieval_scheme,
                        margin=margin_scheme,
                        use_shift_embeds=self.use_shift_embeds,
                        use_gpu=use_gpu,
                        mode='mine',
                        threshold=float('-inf'))
            logger.info("Finish computing the similarity scores, "
                        f"and save at {cand2score_file}.")

        if threshold is not None:
            logger.info(f"Use given threshold {threshold} for retrieval.")

        if self.task_name == 'bucc':
            results = bucc_eval(cand2score_file, gold_file,
                                src_text_file, trg_text_file,
                                src_ind_file, trg_ind_file,
                                predict_file,
                                threshold=threshold)
        elif self.task_name in ('tatoeba',):
            results = generic_eval(cand2score_file,
                                   src_text_file, trg_text_file,
                                   predict_file,
                                   threshold=float('-inf'))

        logger.info(f"--Candidates: {cand2score_file}")
        logger.info(' '.join("{}={:.4f}".format(k, v)
                             for k, v in results.items()))
        return results
