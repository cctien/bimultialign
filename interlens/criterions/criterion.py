from allennlp.common import Registrable
import torch


class Criterion(torch.nn.Module, Registrable):
    """
    A `Criterion` is a `Module` that ...
    """

    def __init__(self,
                 reduction: str = 'mean',
                 verbose: bool = False,) -> None:
        super().__init__()

        self.reduction = reduction
        if reduction == 'mean':
            self._average = torch.mean
        elif reduction == 'sum':
            self._average = torch.sum
        else:
            raise NotImplementedError

        self._verbose = verbose

    def _forward_verbose(self) -> None:
        raise NotImplementedError

# Losses = Dict[str, Dict[str, Union[float, Loss]]]
