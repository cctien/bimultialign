from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from allennlp.common import Lazy, Params
from allennlp.common.checks import ConfigurationError
from allennlp.training.optimizers import Optimizer, make_parameter_groups
import torch

from overrides import overrides


@Optimizer.register('multiplayer', constructor='from_partial_objects')
class MultiplayerOptimizer(Optimizer):
    """
    Registered as an `Optimizer` with name "multi".
    """

    def __init__(self,
                 optimizers: Dict[str, torch.optim.Optimizer],
                 nums_steps: Dict[str, int],
                 ) -> None:

        assert list(optimizers.keys()) == list(nums_steps.keys())
        assert isinstance(optimizers, OrderedDict)
        assert isinstance(nums_steps, OrderedDict)

        defaults = {k: v.defaults for (k, v) in optimizers.items()}
        params = []
        for optimizer in optimizers.values():
            for param_group in optimizer.param_groups:
                params.extend([p for p in param_group['params']])

        super().__init__(params=params, defaults=defaults)

        self.optimizers = optimizers
        self.nums_steps = nums_steps

    @overrides
    def step(self, closure=None,) -> None:
        raise NotImplementedError

    @overrides
    def zero_grad(self,) -> None:
        raise NotImplementedError

    @classmethod
    def from_partial_objects(cls,
                             model_parameters: List[Tuple[str, torch.nn.Parameter]],
                             player_names: List[str],
                             nums_steps: Dict[str, int],
                             optimizers: Dict[str, Lazy[Optimizer]],
                             ) -> 'Optimizer':

        if not set(player_names) == set(optimizers.keys()):
            raise ConfigurationError("`set(player_names)` should be the same as `set(optimizers.keys()). "
                                     f"Got {player_names} and {optimizers.keys()} respectively instead.")

        _optimizers = OrderedDict.fromkeys(player_names)
        for player in player_names:
            _name = '_' + player
            _player_parameters = [(n, p,) for n, p in model_parameters
                                  if hasattr(p, _name)]
            _optimizers[player] = optimizers[player].construct(
                model_parameters=_player_parameters)

        _nums_steps = OrderedDict.fromkeys(player_names)
        for player in player_names:
            _nums_steps[player] = nums_steps[player]

        return cls(optimizers=_optimizers, nums_steps=_nums_steps)
