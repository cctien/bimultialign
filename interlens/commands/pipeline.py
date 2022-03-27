import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Union

from allennlp.commands.subcommand import Subcommand
from allennlp.common import Params
from overrides import overrides

from interlens.commands.evaluate_tasks import get_eval_tasks, run_evaluation_tasks
from interlens.commands.util import (collect_results,
                                     update_config_dict_for_training)

logger = logging.getLogger(__name__)


@Subcommand.register('pipeline')
class Pipeline(Subcommand):
    @overrides
    def add_subparser(self,
                      parser: argparse._SubParsersAction,
                      ) -> argparse.ArgumentParser:
        description = """Run the pipeline experiments with specified lens and other hyperparameters."""
        subparser = parser.add_parser(self.name, description=description,
                                      help="run the pipeline experiments with specified lens and other hyperparameters")

        # Required named arguments.
        required = subparser.add_argument_group('required named arguments')
        required.add_argument('--lens', type=str, required=True,
                              help="lens type with four components.")

        # Optional arguments.

        # Of Paths
        subparser.add_argument('--param_dir', type=str, default='experiments',
                               help="path to dir of parameters describing the models")
        subparser.add_argument('--param_path', type=str, default='aligner/para_aligner.jsonnet',
                               help="path to main aligner model config relative to the param_dir")
        subparser.add_argument('-s', '--serialization_dir', type=str, default='outputs',
                               help="directory in which to save the model and its logs")
        subparser.add_argument('--results_dir', type=str, default='results',
                               help="directory in which to save the metrics and results")
        subparser.add_argument('--experiment_name', type=str, default='',
                               help="additional string attached to the name of the experiment.")

        # About the Model
        subparser.add_argument('--pivot_languages', type=str, default='de fr ru zh',
                               help="codenames for the pivot language other than English")
        subparser.add_argument('--pivot_pair', type=str, default=None,
                               help="provide one pair (sep by space) of pivot languages")
        subparser.add_argument('--criterion_variants', type=str, default='neg_sampling noise_contrastive triplet_ranking',
                               help="Variants of criterions for the aligner")
        subparser.add_argument('--nums_random_samples', type=str, default='0',
                               help="numbers of random samples for contrastive loss of the alinger")
        subparser.add_argument('--margins', type=str, default='',
                               help="the values of margins to be experimented with.")
        # About the Model (specific to the GAN aligner)
        subparser.add_argument('--critic_criterion', type=str, default=None,
                               help="Variant of criterions for the aligner")
        subparser.add_argument('--cycle_loss_lambda', type=float, default=None,
                               help="Multiplier before the cycle loss for the generator")

        # About Training
        subparser.add_argument('--num_epochs', type=int, default=None,
                               help="number of epochs for training the aligner model")
        subparser.add_argument('--learning_rate', type=float, default=None,
                               help="If given, learning rate for the optimizer to override the config")
        subparser.add_argument('--batch_size', type=int, default=None,
                               help="If non-empty, the batch size to override config.")
        subparser.add_argument('--epoch_size', type=int, default=None,
                               help="If non-empty, the epoch size (number of samples in an epoch) to override config.")
        subparser.add_argument('--patience', type=int, default=None,
                               help="If non-empty, the patience to override config.")
        subparser.add_argument('--cuda_device', type=int, default=None,
                               help="id of GPU to use")
        subparser.add_argument('--not_use_amp', action='store_true', default=False,
                               help="choose not to use amp")
        # subparser.add_argument('-r', '--recover', action='store_true', default=False,
        #                        help="recover training from the state in serialization_dir",)
        # subparser.add_argument('--test', action='store_true', default=False,
        #                        help="Also evaluate on test data",)

        # About Training (specific to the GAN aligner)
        subparser.add_argument('--critic_num_steps', type=int, default=None,
                               help="Number of backprop steps of the critic module relative to that of the generator")

        subparser.set_defaults(func=pipeline_from_args)

        return subparser


def pipeline_from_args(args: argparse.Namespace,) -> None:
    if args.pivot_pair is not None:
        pivot_pair = args.pivot_pair.split()
    else:
        pivot_pair = args.pivot_pair

    pipeline_from_files(pivot_languages=args.pivot_languages.split(),
                        pivot_pair = pivot_pair,
                        lens=args.lens.split(),
                        param_dir=Path(args.param_dir),
                        param_path=Path(args.param_path),
                        serialization_dir=Path(args.serialization_dir),
                        results_dir=Path(args.results_dir),
                        experiment_name=args.experiment_name,
                        criterion_variants=args.criterion_variants.split(),
                        nums_random_samples=[int(n)
                                             for n in args.nums_random_samples.split()],
                        margins=[get_margin_value(m)
                                 for m in args.margins.split()],
                        critic_criterion=args.critic_criterion,
                        cycle_loss_lambda=args.cycle_loss_lambda,
                        learning_rate=args.learning_rate,
                        num_epochs=args.num_epochs,
                        batch_size=args.batch_size,
                        epoch_size=args.epoch_size,
                        patience=args.patience,
                        cuda_device=args.cuda_device,
                        not_use_amp=args.not_use_amp,
                        # recover=args.recover,
                        critic_num_steps=args.critic_num_steps)


def pipeline_from_files(pivot_languages: List[str],
                        pivot_pair: Optional[List[str]],
                        lens: List[str],
                        param_dir: Path,
                        param_path: Path,
                        serialization_dir: Path,
                        results_dir: Path,
                        experiment_name: str,
                        criterion_variants: List[str],
                        nums_random_samples: List[int],
                        margins: List[Union[float, None]],
                        critic_criterion: Optional[str],
                        cycle_loss_lambda: Optional[float],
                        learning_rate: Optional[float],
                        num_epochs: Optional[int],
                        batch_size: Optional[int],
                        epoch_size: Optional[int],
                        patience: Optional[int],
                        cuda_device: Optional[int],
                        not_use_amp: bool,
                        # recover: bool,
                        critic_num_steps: Optional[int],
                        ) -> None:

    time_sig = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_(%z)')

    lens_dir = param_dir.joinpath('lens')
    criterion_dir = param_dir.joinpath('criterion')

    trial_name = f"{param_path.stem}_{get_lens_name(lens)}"
    if param_path.stem == 'baseline':
        no_pretrained_lens = True
    else:
        no_pretrained_lens = False

    if experiment_name:
        trial_name += f"_{experiment_name}"

    master_output_dir = serialization_dir.joinpath(trial_name)
    results_dir = results_dir.joinpath(trial_name)
    temp_dir = Path('tmp/config_files')
    for directory in (master_output_dir, results_dir, temp_dir,):
        directory.mkdir(parents=True, exist_ok=True)

    config_dict = Params.from_file(param_dir.joinpath(param_path)
                                   ).as_dict()

    # Training configuration.
    config_dict = update_config_dict_for_training(config_dict=config_dict,
                                                  learning_rate=learning_rate,
                                                  batch_size=batch_size,
                                                  epoch_size=epoch_size,
                                                  patience=patience,
                                                  cuda_device=cuda_device,
                                                  not_use_amp=not_use_amp,
                                                  num_epochs=num_epochs)
    if critic_num_steps is not None:
        config_dict['trainer']['optimizer']['nums_steps']['critic'] = critic_num_steps

    # The aligner model configuration.

    lens_components = {'layer_pooler': lens[0],
                       'seq_encoder': lens[1],
                       'vec_encoder': lens[2],
                       'feedforward': lens[3], }
    lens_dict = Params.from_file(lens_dir.joinpath('basic_lens.libsonnet')
                                 ).as_dict()
    for comp_name, comp in lens_components.items():
        if comp == 'none':
            lens_dict[comp_name] = None
        else:
            comp_path = lens_dir.joinpath(f"{comp_name}/{comp}.libsonnet")
            lens_dict[comp_name] = Params.from_file(comp_path).as_dict()

    config_dict['model']['lens'] = lens_dict

    if critic_criterion is not None:
        critic_criterion_path = criterion_dir.joinpath(f"{critic_criterion}_"
                                                       "criterion.libsonnet")
        config_dict['model']['critic_criterion'] = Params.from_file(
            critic_criterion_path).as_dict()

    if cycle_loss_lambda is not None:
        config_dict['model']['aggregate_criterion']['lamda_weights'] = [
            cycle_loss_lambda, ]

    # The evaluation models.
    eval_tasks = get_eval_tasks(param_dir)

    if no_pretrained_lens:

        rev_config_file = temp_dir.joinpath(f"baseline_{get_lens_name(lens)}_"
                                            f"{time_sig}.json")
        rev_config_file.parent.mkdir(parents=True, exist_ok=True)
        rev_config_file.write_text(json.dumps(config_dict, indent=4))

        baseline_program = ['python', 'interlens', 'train',
                            str(rev_config_file),
                            '-s',  str(master_output_dir), ]
        subprocess.run(baseline_program)
        run_evaluation_tasks(eval_tasks=eval_tasks,
                             output_dir=master_output_dir,
                             cuda_device=cuda_device,
                             no_pretrained_lens=no_pretrained_lens,
                             )
        collect_results(master_output_dir, results_dir)
        return

    # Allow empty string as input for args.margins.
    if len(margins) == 0:
        margins = [None, ]

    # Iterate over the hyperparameters of criterions.
    for criterion in criterion_variants:

        criterion_path = criterion_dir.joinpath(f"{criterion}_criterion"
                                                ".libsonnet")
        criterion_dict = Params.from_file(criterion_path).as_dict()

        for margin in margins:
            criterion_dict['margin'] = margin

            for num_random_samples in nums_random_samples:

                criterion_dict['num_random_samples'] = num_random_samples
                if criterion_dict['type'] is None:
                    criterion_dict = None

                config_dict['model']['criterion'] = criterion_dict

                if pivot_pair is None:
                    for pivot_lang in pivot_languages:

                        config_dict['dataset_reader']['pair_names'] = [pivot_lang,
                                                                       'en', ]

                        model_name = get_model_name(lens, criterion_dict,
                                                    pivot_lang)
                        rev_model_file = temp_dir.joinpath(f"{model_name}_{time_sig}_"
                                                        "config.json")
                        rev_model_file.write_text(json.dumps(config_dict,
                                                            indent=4))
                        model_output_dir = master_output_dir.joinpath(model_name)

                        aligner_program = ['python', 'interlens', 'train',
                                        str(rev_model_file),
                                        '-s',  str(model_output_dir), ]
                        subprocess.run(aligner_program)

                        run_evaluation_tasks(eval_tasks=eval_tasks,
                                            output_dir=model_output_dir,
                                            cuda_device=cuda_device,
                                            )

                else:
                    config_dict['dataset_reader']['pair_names'] = pivot_pair

                    model_name = get_model_name(lens, criterion_dict,
                                                ''.join(pivot_pair))
                    rev_model_file = temp_dir.joinpath(f"{model_name}_{time_sig}_"
                                                       "config.json")
                    rev_model_file.write_text(json.dumps(config_dict,
                                                        indent=4))
                    model_output_dir = master_output_dir.joinpath(model_name)

                    aligner_program = ['python', 'interlens', 'train',
                                       str(rev_model_file),
                                       '-s',  str(model_output_dir), ]
                    subprocess.run(aligner_program)

                    run_evaluation_tasks(eval_tasks=eval_tasks,
                                         output_dir=model_output_dir,
                                         cuda_device=cuda_device,
                                         )

    collect_results(master_output_dir, results_dir)
    return


def get_margin_value(margin: Union[str, None],
                     ) -> Union[float, None]:

    try:
        return float(margin)
    except ValueError:
        return None


def get_lens_name(lens: List[str],):
    assert len(lens) == 4
    return '_'.join(lens)


def get_model_name(lens: List[str],
                   criterion_dict: Dict[str, Any],
                   pivot_lang: str, ) -> str:

    criterion = criterion_dict['type']
    num_random_samples = criterion_dict['num_random_samples']
    margin = criterion_dict['margin']
    lens_name = get_lens_name(lens)
    return (f"{lens_name}_{criterion}_{num_random_samples}_mg{margin}_{pivot_lang}")
