from typing import Optional

from allennlp.commands import parse_args
from allennlp.common.plugins import import_plugins
from allennlp.common.util import import_module_and_submodules

from interlens.commands.evaluate_tasks import EvaluateTasks
from interlens.commands.pipeline import Pipeline
from interlens.commands.retrieve import Retrieve


def main(prog: Optional[str] = None,) -> None:
    """
    With this custom function, the module `interlens` is included in the list of `include_package'.
    Below are original docstrings from [allennlp.commands](https://github.com/allenai/allennlp/blob/master/allennlp/commands/__init__.py).
    The [`run`](./train.md#run) command only knows about the registered classes in the ``allennlp``
    codebase. In particular, once you start creating your own `Model` s and so forth, it won't
    work for them, unless you use the ``--include-package`` flag or you make your code available
    as a plugin (see [`plugins`](./plugins.md)).
    """

    import_plugins()

    parser, args = parse_args(prog)

    # Include this package, in addition to any other packages passed.
    args.include_package.append('interlens')

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if 'func' in dir(args):
        # Import any additional modules needed (to register custom classes).
        for package_name in getattr(args, 'include_package', []):
            import_module_and_submodules(package_name)
        args.func(args)
    else:
        parser.print_help()
