import argparse
from pathlib import Path
import subprocess
from typing import Dict, List, Optional

from allennlp.commands.subcommand import Subcommand
from overrides import overrides

from interlens.commands.util import collect_results


@Subcommand.register('evaluate_tasks')
class EvaluateTasks(Subcommand):
    @overrides
    def add_subparser(self,
                      parser: argparse._SubParsersAction,
                      ) -> argparse.ArgumentParser:
        description = """Run several evaluation tasks given a pretrained model."""
        subparser = parser.add_parser(self.name, description=description,
                                      help="Run several evaluation tasks given a pretrained model")

        # Required named arguments.
        required = subparser.add_argument_group('required named arguments')
        required.add_argument('--model_dir', type=str,
                              help="directory where the (pretrained) model archive is")

        # Optional arguments.

        # Of Paths
        subparser.add_argument('-s', '--serialization_dir', type=str, default='outputs',
                               help="directory in which to save the model and its logs")
        subparser.add_argument('--param_dir', type=str, default='experiments',
                               help="path to dir of parameters describing the models")
        subparser.add_argument('--results_dir', type=str, default='results',
                               help="directory in which to save the metrics and results")

        # About the Model
        subparser.add_argument('--no_pretrained_lens', action='store_true', default=False,
                               help="not using pretrained lens, for baselines.")

        # About Training
        subparser.add_argument('--cuda_device', type=int, default=None,
                               help="id of GPU to use")

        subparser.set_defaults(func=evaluate_tasks_from_args)

        return subparser


def evaluate_tasks_from_args(args: argparse.Namespace,) -> None:

    evaluate_tasks_from_files(model_dir=Path(args.model_dir),
                              serial_dir=Path(args.serialization_dir),
                              param_dir=Path(args.param_dir),
                              results_dir=Path(args.results_dir),
                              no_pretrained_lens=args.no_pretrained_lens,
                              cuda_device=args.cuda_device,
                              )


def evaluate_tasks_from_files(model_dir: Path,
                              serial_dir: Path,
                              param_dir: Path,
                              results_dir: Path,
                              no_pretrained_lens: bool,
                              cuda_device: Optional[int],
                              ) -> None:

    eval_tasks = get_eval_tasks(param_dir=param_dir)
    run_evaluation_tasks(eval_tasks=eval_tasks,
                         output_dir=model_dir,
                         cuda_device=cuda_device,
                         no_pretrained_lens=no_pretrained_lens,
                         )

    results_dir = results_dir.joinpath(model_dir.relative_to(serial_dir))
    collect_results(outputs_dir=model_dir, results_dir=results_dir)


def get_eval_tasks(param_dir: Path,) -> Dict[str, List[Path]]:

    retrieve_dir = param_dir.joinpath('retrieval')
    eval_tasks = {
        'retrieve': [
            # retrieve_dir.joinpath('bucc_cosine_n4.jsonnet'),  # Uncomment this line to evaluate on BUCC
            retrieve_dir.joinpath('tatoeba_cosine_n4.jsonnet'),
        ],
        }

    return eval_tasks


def run_evaluation_tasks(eval_tasks: Dict[str, List[Path]],
                         output_dir: Path,
                         cuda_device: Optional[int],
                         no_pretrained_lens: bool = False,
                         ) -> None:

    for eval_cmd, task_config_files in eval_tasks.items():
        for eval_config in task_config_files:

            program = ['python', 'interlens', eval_cmd,
                       str(eval_config),
                       '-s',  str(output_dir), ]

            if no_pretrained_lens and eval_cmd == 'retrieve':
                program.extend(['--pivot_src', 'none', ])

            if cuda_device is not None:
                program.extend(['--cuda_device', str(cuda_device), ])

            subprocess.run(program)

