from allennlp.common import Registrable
from allennlp.nn.activations import Activation, _ActivationLambda
import torch


class SoftmaxLastdim(torch.nn.Softmax):
    def __init__(self,):
        super().__init__(dim=-1)


class LogSoftmaxLastdim(torch.nn.LogSoftmax):
    def __init__(self,):
        super().__init__(dim=-1)


registered_activations = Registrable._registry[Activation]
additional_activateions = {
    # Just register with a name more explicit on what sigmoid function is used, i.e., the logistic.
    # Having been used for JS-divergence optimization in [DIM](https://arxiv.org/abs/1808.06670).
    # Also formulated as the activation for the GAN loss in [f-GAN](https://arxiv.org/abs/1606.00709).
    # 'gelu': (
    #     torch.nn.GELU,
    #     None,),
    'log_logistic': (
        torch.nn.LogSigmoid,
        None,),
    'softmax_lastdim': (
        SoftmaxLastdim,
        None,),
    'log_softmax_lastdim': (
        LogSoftmaxLastdim,
        None,)
    # # `'reverse_softplus'` is mathmatically identical to `'log_logistic'`.
    # 'reverse_softplus': (
    #     lambda: _ActivationLambda(
    #         lambda x: -torch.nn.functional.softplus(-x), 'ReverseSoftplus'),
    #     None,),
}
for key in additional_activateions:
    assert key not in registered_activations, f"`Activation` with name `{key}` already registered."

registered_activations.update(additional_activateions)
