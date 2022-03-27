"""
The `retrieve` subcommand can be used to

Parts of the code from [allennlp](https://github.com/allenai/allennlp/blob/master/allennlp/commands/evaluate.py),
and from [xtreme](https://github.com/google-research/xtreme/blob/master/scripts/run_bucc2018.sh).
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from allennlp.commands.subcommand import Subcommand
from allennlp.common import Params
from allennlp.common.logging import prepare_global_logging
from allennlp.data import DatasetReader
import numpy
from overrides import overrides

from interlens.commands.util import dump_metrics, get_embeddings
from interlens.evaluation.retriever import Retriever

logger = logging.getLogger(__name__)


@Subcommand.register('retrieve')
class Retrieve(Subcommand):
    @overrides
    def add_subparser(self,
                      parser: argparse._SubParsersAction,
                      ) -> argparse.ArgumentParser:
        description = """Retrieve based on the specified model + dataset."""
        subparser = parser.add_parser(self.name, description=description,
                                      help="retrieve based on model and dataset.")

        # Required arguments.
        subparser.add_argument('param_path', type=str,
                               help="path(s) to parameter file describing the retriever model.")

        # Required named arguments.
        required = subparser.add_argument_group('required named arguments')
        required.add_argument('-s', '--serialization_dir', type=str, required=True,
                              help="directory in which to save the model and its logs")

        # Optional arguments.
        subparser.add_argument('--cuda_device', type=int, default=0,
                               help="id of GPU to use for extracting embeddings")
        subparser.add_argument('--batch_size', type=int, default=None,
                               help="the batch size to use during embedding if not empty")
        subparser.add_argument('-o', '--overrides', type=str, default='',
                               help="a Jsonnet structure used to override the experiment configuration of the retriever")
        subparser.add_argument('--use_gpu_search', action='store_true',
                               help="Use GPU for KNN search.")
        subparser.add_argument('--test', action='store_true',
                               help="also evaluate on the test split if passed")
        subparser.add_argument('--pivot_src', type=str, default=None,
                               help="source language code the optimized threshold from which to be experimented with.")
        subparser.add_argument('--pivot_trg', type=str, default='en', choices=['en', ],
                               help="trg language code the optimized threshold from which to be experimented with.")
        subparser.add_argument('--threshold', type=float, default=None,
                               help="the threshold for bitext mining. "
                               "Default to `None`, and the threshold will be searched.")
        subparser.add_argument('--experiment_name', type=str, default='',
                               help="additional string attached to the name of the experiment.")
        # subparser.add_argument('--splits', type=str, default=None,
        #                        help="Data splits of the retrieval task is passed.")

        subparser.set_defaults(func=retrieve_from_args)

        return subparser


def retrieve_from_args(args: argparse.Namespace,) -> None:

    param_paths = args.param_path.split()
    for param_path in param_paths:
        experiment_name = Path(param_path).stem
        if args.experiment_name:
            experiment_name += f"_{args.experiment_name}"

        output_dir = Path(args.serialization_dir).joinpath(experiment_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        prepare_global_logging(output_dir)

        encoder_dir = Path(args.serialization_dir)
        encoder_archive = encoder_dir.joinpath('model.tar.gz')
        encoder_config = Params.from_file(encoder_dir.joinpath('config.json'))

        # Get the experiment configuration from the parameter file.
        config = Params.from_file(param_path, args.overrides)
        config.to_file(output_dir.joinpath('config.json'))
        data_config = config.pop('dataset_reader')
        data_dir = data_config.get('data_dir')
        langs = data_config.get('all_lang_names')
        dataset_reader = DatasetReader.from_params(data_config)
        retriever = Retriever.from_params(config.pop('retriever'))
        zero_threshold = config.pop('zero_threshold', default=False)
        _batch_size = config.pop('batch_size')
        batch_size = args.batch_size or _batch_size
        splits = config.pop('splits')
        config.assert_empty('Retrieve')
        if args.test and splits == ['dev', ]:
            splits.append('test')

        pair_names = encoder_config.pop('dataset_reader').pop('pair_names',
                                                              default=None)
        threshold_pivots = None
        if not zero_threshold:
            if args.pivot_src not in (None, 'none',):
                threshold_pivots = {'src': args.pivot_src,
                                    'trg': args.pivot_trg, }
            elif pair_names is not None:
                pair_names.remove(args.pivot_trg)  # Remove 'en'.
                assert len(pair_names) == 1
                if pair_names[0] in ('de', 'fr', 'ru', 'zh',):
                    threshold_pivots = {'src': pair_names[0],
                                        'trg': args.pivot_trg, }

        # Get the metrics on the retrieval task.
        file_paths = get_file_paths(experiment_name,
                                    output_dir,
                                    data_dir,
                                    langs,
                                    splits)
        retrieve_from_files(src_langs=langs['src'],
                            trg_langs=langs['trg'],
                            threshold_pivots=threshold_pivots,
                            file_paths=file_paths,
                            encoder_archive=encoder_archive,
                            dataset_reader=dataset_reader,
                            retriever=retriever,
                            batch_size=batch_size,
                            cuda_device=args.cuda_device,
                            splits=splits,
                            threshold=args.threshold,
                            use_gpu_search=args.use_gpu_search,
                            zero_threshold=zero_threshold)


def retrieve_from_files(src_langs: List[str],
                        trg_langs: List[str],
                        threshold_pivots: Optional[Dict[str, str]],
                        file_paths: Dict[str, Dict[str, Dict[str, Path]]],
                        encoder_archive: str,
                        dataset_reader: DatasetReader,
                        retriever: Retriever,
                        batch_size: int,
                        cuda_device: int,
                        splits: List[str],
                        threshold: Optional[float] = None,
                        use_gpu_search: bool = False,
                        zero_threshold: bool = False,
                        ) -> None:

    margin_schemes = ('absolute', 'difference', 'ratio')
    retrieval_schemes = ('fwd', 'bwd', 'max')
    margin_schemes_test = ('difference', 'ratio')
    retrieval_schemes_test = ('max',)

    results_all = {spl: {src: {trg: {retrieval: {margin: {}
                                                 for margin in margin_schemes}
                                     for retrieval in retrieval_schemes}
                               for trg in trg_langs}
                         for src in src_langs}
                   for spl in splits}

    interim_dir = file_paths['interim_dir']

    for spl in splits:

        if threshold_pivots is not None:
            results_all['pivot_languages'] = threshold_pivots
            src_langs.sort(key=lambda x: x != threshold_pivots['src'])
            trg_langs.sort(key=lambda x: x != threshold_pivots['trg'])
            logger.info(f"Perform retrieval tasks with source languages {src_langs} "
                        f"and target languages {trg_langs}, in this order.")
            got_pivoted_thresholds = False
        else:
            got_pivoted_thresholds = None

        pivoted_threshold = None

        if spl == 'test':
            _retrieval_schemes = retrieval_schemes_test
            _margin_schemes = margin_schemes_test
        else:
            _retrieval_schemes = retrieval_schemes
            _margin_schemes = margin_schemes

        for src in src_langs:
            for trg in trg_langs:

                text_files = file_paths['text'][src][trg][spl]
                embedding_files = file_paths['embeddings'][src][trg][spl]
                embeddings = get_embeddings(archive_file=encoder_archive,
                                            dataset_reader=dataset_reader,
                                            src=src,
                                            trg=trg,
                                            text_files=text_files,
                                            embedding_files=embedding_files,
                                            batch_size=batch_size,
                                            cuda_device=cuda_device)

                for retrieval in _retrieval_schemes:
                    for margin in _margin_schemes:

                        if got_pivoted_thresholds:
                            pivoted_threshold = (results_all[spl]
                                                 [threshold_pivots['src']]
                                                 [threshold_pivots['trg']]
                                                 [retrieval][margin]
                                                 ['optimized_threshold']
                                                 ['threshold'])

                        results = retrieve_specific(embeddings=embeddings,
                                                    text_files=text_files,
                                                    interim_dir=interim_dir,
                                                    src=src,
                                                    trg=trg,
                                                    spl=spl,
                                                    retrieval=retrieval,
                                                    margin=margin,
                                                    retriever=retriever,
                                                    pivoted_threshold=pivoted_threshold,
                                                    given_threshold=threshold,
                                                    use_gpu=use_gpu_search,
                                                    zero_threshold=zero_threshold)
                        results_all[spl][src][trg][retrieval][margin] = results

                if got_pivoted_thresholds is False:
                    logger.info(f"Texts from the pivot language pairs {src}-{trg} are mined, "
                                "and the pivot thresholds optimzed over the pair are got.")
                    got_pivoted_thresholds = True

    dump_metrics(file_paths['output_dir'].joinpath('metrics.json'),
                 results_all,
                 log=True)


def retrieve_specific(embeddings: numpy.ndarray,
                      text_files: Dict[str, Path],
                      interim_dir: Path,
                      src: List[str],
                      trg: List[str],
                      spl: str,
                      retrieval: 'str',
                      margin: 'str',
                      retriever: Retriever,
                      pivoted_threshold: Optional[float] = None,
                      given_threshold: Optional[float] = None,
                      use_gpu: bool = False,
                      zero_threshold: bool = False,
                      ) -> Dict[str, Any]:

    prefix = f"{src}-{trg}.{spl}.{retrieval}retrieval.{margin}margin"
    cand2score_file = interim_dir.joinpath(f"cand2score-{prefix}.tsv")
    predict_file = interim_dir.joinpath(f"predict-{prefix}.tsv")

    results = {}

    if zero_threshold:
        results['zero_threshold'] = retriever.retrieve(
            embeddings[src],
            embeddings[trg],
            **text_files,
            cand2score_file=cand2score_file,
            predict_file=predict_file,
            retrieval_scheme=retrieval,
            margin_scheme=margin,
            threshold=float('-inf'),
            use_gpu=use_gpu)
        return results

    results['optimized_threshold'] = retriever.retrieve(
        embeddings[src],
        embeddings[trg],
        **text_files,
        cand2score_file=cand2score_file,
        predict_file=predict_file,
        retrieval_scheme=retrieval,
        margin_scheme=margin,
        threshold=None,
        use_gpu=use_gpu)

    if pivoted_threshold is not None:
        results['pivoted_threshold'] = retriever.retrieve(
            embeddings[src],
            embeddings[trg],
            **text_files,
            cand2score_file=cand2score_file,
            retrieval_scheme=retrieval,
            margin_scheme=margin,
            threshold=pivoted_threshold,
            use_gpu=use_gpu)

    if given_threshold is not None:
        results['given_threshold'] = retriever.retrieve(
            embeddings[src],
            embeddings[trg],
            **text_files,
            cand2score_file=cand2score_file,
            retrieval_scheme=retrieval,
            margin_scheme=margin,
            threshold=given_threshold,
            use_gpu=use_gpu)

    return results


def get_file_paths(experiment_name: str,
                   output_dir: Path,
                   data_dir: Union[Path, str],
                   langs: Dict[str, List[str]],
                   splits: List[str],
                   ) -> Dict[str, Dict[str, Dict[str, Path]]]:

    affix0 = experiment_name.split('_')[0]
    if affix0 == 'tatoeba':
        splits = ['', ]

    embeddings_dir = output_dir.parent.joinpath('embeddings_retrieval_'
                                                f'{affix0}')

    interim_dir = output_dir.joinpath('interim')
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(data_dir)

    file_paths = {subdir: {src: {trg: {spl: {}
                                       for spl in splits}
                                 for trg in langs['trg']}
                           for src in langs['src']}
                  for subdir in ('text', 'embeddings',)}
    file_paths['output_dir'] = output_dir
    file_paths['interim_dir'] = interim_dir
    file_paths['splits'] = splits

    for src in langs['src']:
        for trg in langs['trg']:
            for spl in splits:

                if affix0 == 'tatoeba':
                    prefix = f"{affix0}.{src}-{trg}"
                else:
                    prefix = f"{src}-{trg}.{spl}"

                text_files = file_paths['text'][src][trg][spl]
                if affix0 == 'tatoeba':
                    text_files['src_text_file'] = data_dir.joinpath(
                        f"{prefix}.{src}")
                    text_files['trg_text_file'] = data_dir.joinpath(
                        f"{prefix}.{trg}")
                    text_files['src_ind_file'] = None
                    text_files['trg_ind_file'] = None
                    text_files['gold_file'] = None
                else:
                    text_files['src_text_file'] = data_dir.joinpath(
                        f"{prefix}.{src}.txt")
                    text_files['trg_text_file'] = data_dir.joinpath(
                        f"{prefix}.{trg}.txt")
                    text_files['src_ind_file'] = data_dir.joinpath(
                        f"{prefix}.{src}.id")
                    text_files['trg_ind_file'] = data_dir.joinpath(
                        f"{prefix}.{trg}.id")
                    text_files['gold_file'] = data_dir.joinpath(
                        f"{prefix}.gold")

                emb_files = file_paths['embeddings'][src][trg][spl]
                emb_files['src_emb_file'] = embeddings_dir.joinpath(
                    f"{prefix}.{src}.emb.npy")
                emb_files['trg_emb_file'] = embeddings_dir.joinpath(
                    f"{prefix}.{trg}.emb.npy")

    return file_paths
