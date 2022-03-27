import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Union

from allennlp.common.checks import check_for_gpu
from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data import (AllennlpDataset, AllennlpLazyDataset,
                           DataLoader, DatasetReader)
from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.nn.util import move_to_device
from allennlp.training.util import evaluate
import numpy
import torch

logger = logging.getLogger(__name__)


def collect_results(outputs_dir: Union[Path, str] = 'outputs',
                    results_dir: Union[Path, str] = 'results',) -> None:

    outputs_dir = Path(outputs_dir)
    results_dir = Path(results_dir)

    for pattern in ('*.json', '*.log', 'events.out.tfevents.*',):
        for output_file in outputs_dir.rglob(pattern):

            file_relative_path = output_file.relative_to(outputs_dir)
            result_file = results_dir.joinpath(file_relative_path)

            result_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_file, result_file)


def update_config_dict_for_training(config_dict: Dict[str, Any],
                                    learning_rate: Optional[float],
                                    batch_size: Optional[int],
                                    epoch_size: Optional[int],
                                    patience: Optional[int],
                                    cuda_device: Optional[int],
                                    not_use_amp: bool,
                                    num_epochs: Optional[int],
                                    ) -> Dict[str, Any]:

    if learning_rate is not None:
        config_dict['trainer']['optimizer']['lr'] = learning_rate

    if batch_size is not None:
        if 'batch_sampler' in config_dict['data_loader']:
            config_dict['data_loader']['batch_sampler']['batch_size'] = batch_size
        else:
            config_dict['data_loader']['batch_size'] = batch_size

    if epoch_size is not None:
        if 'batch_sampler' in config_dict['data_loader']:
            _batch_size = config_dict['data_loader']['batch_sampler']['batch_size']
        else:
            _batch_size = config_dict['data_loader']['batch_size']

        batches_per_epoch: int = ((epoch_size - 1) // _batch_size) + 1
        config_dict['data_loader']['batches_per_epoch'] = batches_per_epoch

    if patience is not None:
        config_dict['trainer']['patience'] = patience

    if cuda_device is not None:
        config_dict['trainer']['cuda_device'] = cuda_device
        if cuda_device < 0:
            config_dict['trainer']['use_amp'] = False

    if not_use_amp:
        config_dict['trainer']['use_amp'] = False

    if num_epochs is not None:
        config_dict['trainer']['num_epochs'] = num_epochs

    return config_dict


def dump_metrics(file_path: Optional[str],
                 metrics: Dict[str, Any],
                 log: bool = False,
                 indent: int = 4) -> None:
    """
    Slightly modified from [Allennlp.common.util.dump_metrics](https://github.com/allenai/allennlp/blob/master/allennlp/commands/evaluate.py)
    with customizable indentation in the json file.
    """
    metrics_json = json.dumps(metrics, indent=indent)

    if file_path:
        with open(file_path, 'w') as metrics_file:
            metrics_file.write(metrics_json)

    if log:
        logger.info(f"Metrics: {metrics_json}")


def evaluate_from_file(archive_file: Union[Path, str],
                       input_file: Union[Path, str],
                       overrides: str = '',
                       cuda_device: Union[int, List[int]] = -1,
                       batch_size: Optional[int] = None,
                       ) -> Dict[str, float]:
    """
    Slightly modified from [Allennlp.commands.evaluate](https://github.com/allenai/allennlp/blob/master/allennlp/commands/evaluate.py).
    """

    archive = load_archive(archive_file,
                           cuda_device=cuda_device,
                           overrides=overrides)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    validation_dataset_reader_params = config.pop(
        'validation_dataset_reader', None)
    if validation_dataset_reader_params is not None:
        dataset_reader = DatasetReader.from_params(
            validation_dataset_reader_params)
    else:
        dataset_reader = DatasetReader.from_params(
            config.pop('dataset_reader'))

    evaluation_data_path = input_file
    logger.info(f"Reading evaluation data from {evaluation_data_path}")

    instances = dataset_reader.read(evaluation_data_path)
    instances.index_with(model.vocab)

    data_loader_params = config.pop("validation_data_loader", None)
    if data_loader_params is None:
        data_loader_params = config.pop("data_loader")

    if batch_size:
        data_loader_params['batch_size'] = batch_size

    data_loader = DataLoader.from_params(dataset=instances,
                                         params=data_loader_params)

    metrics = evaluate(model, data_loader, cuda_device)
    logger.info("Finished evaluating.")
    dump_metrics(None, metrics, log=True)

    return metrics


def get_embeddings(archive_file: Union[Path, str],
                   dataset_reader: DatasetReader,
                   src: str,
                   trg: str,
                   text_files: Dict[str, Path],
                   embedding_files: Dict[str, Path],
                   batch_size: int = 64,
                   cuda_device: int = -1
                   ) -> Dict[str, numpy.ndarray]:

    langs = (src, trg,)
    text_files = (text_files['src_text_file'],
                  text_files['trg_text_file'],)
    embedding_files = (embedding_files['src_emb_file'],
                       embedding_files['trg_emb_file'],)
    embeddings = {}

    for i, lang in enumerate(langs):
        text_file = text_files[i]
        emb_file = embedding_files[i]
        if emb_file.is_file():
            embeddings[lang] = numpy.load(emb_file)
        else:
            embeddings[lang] = extract_embeddings(archive_file=archive_file,
                                                  input_file=text_file,
                                                  dataset_reader=dataset_reader,
                                                  batch_size=batch_size,
                                                  cuda_device=cuda_device)
            numpy.save(emb_file, embeddings[lang])
            torch.cuda.empty_cache()

    return embeddings


def extract_embeddings(archive_file: Union[Path, str],
                       input_file: Union[Path, str],
                       dataset_reader: DatasetReader,
                       # overrides: str = '',
                       batch_size: int = 64,
                       cuda_device: int = -1,
                       use_torch: bool = False,
                       ) -> numpy.ndarray:

    model = get_model_as_encoder(archive_file, cuda_device=cuda_device)

    # Read the data.
    logger.info(f"Extracing embeddings from texts in {input_file}.")
    instances = dataset_reader.read(input_file)
    instances.index_with(model.vocab)
    data_loader_params = Params({'shuffle': False, 'batch_size': batch_size, })
    data_loader = DataLoader.from_params(dataset=instances,
                                         params=data_loader_params)

    # Extract embeddings by iterating over batches.

    num_instances = get_num_instances(instances, dataset_reader)
    if use_torch:
        _cuda_device = None if cuda_device < 0 else cuda_device
        embeddings = torch.zeros((num_instances, model.lens.get_inter_dim()),
                                 dtype=torch.float,
                                 device=_cuda_device)
    else:
        embeddings = numpy.zeros((num_instances, model.lens.get_inter_dim()),
                                 dtype=numpy.float32)

    check_for_gpu(cuda_device)
    with torch.no_grad():
        model.eval()
        iterator = iter(data_loader)
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(iterator)

        batch_ind = 0
        for batch in generator_tqdm:
            start_ind = batch_ind * batch_size
            end_ind = min((batch_ind + 1) * batch_size, num_instances)
            batch_ind += 1

            batch = move_to_device(batch, cuda_device)
            emb_batch = model.get_sentence_embeddings(**batch)

            if use_torch:
                embeddings[start_ind: end_ind] = emb_batch.detach()
            else:
                embeddings[start_ind: end_ind] = (
                    emb_batch.detach().cpu().numpy().astype(numpy.float32))
            if end_ind == num_instances:
                break

    return embeddings


def get_model_as_encoder(archive_file: Union[Path, str],
                         cuda_device: int = -1,
                         ) -> Model:

    archive = load_archive(archive_file,
                           cuda_device=cuda_device,
                           # overrides='{"model.type": "sentence_encoder"}',
                           )
    prepare_environment(archive.config)
    archive.model.eval()

    return archive.model


def get_num_instances(instances: Union[AllennlpDataset, AllennlpLazyDataset],
                      dataset_reader: DatasetReader,
                      ) -> int:

    try:
        num_instances = len(instances)
    except TypeError:
        num_instances = dataset_reader.number_of_instances

    return num_instances
