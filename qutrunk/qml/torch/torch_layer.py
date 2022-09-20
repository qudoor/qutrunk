from typing import Tuple, Any, Optional, cast, Union
import numpy as np

from torch import Tensor, sparse_coo_tensor, einsum
from torch.autograd import Function
from torch.nn import Module, Parameter as TorchParam
from qutrunk.qml.neural_network import NeuralNetwork

class TorchLayer(Module):

    class _TorchNNFunction(Function):
        @staticmethod
        def forward(ctx: Any, input_data: Tensor, neural_network: NeuralNetwork) -> Tensor:
            ctx.neural_network = neural_network
            result = neural_network.forward(input_data)
            ctx.save_for_backward(input_data, result)
            return result

        @staticmethod
        def backward(ctx: Any, grad_output: Tensor) -> Tuple:
            input_data, expectation_z = ctx.saved_tensors
            neural_network = ctx.neural_network

            input_grad = neural_network.backward(input_data)
            if input_grad is not None:
                input_grad = input_grad.float() * grad_output.float()

            return input_grad, None, None

    def __init__(self,neural_network: NeuralNetwork):
        super().__init__()
        self._neural_network = neural_network

    @property
    def neural_network(self) -> NeuralNetwork:
        return self._neural_network

    def forward(self, input_data: Tensor) -> Tensor:
        input_ = input_data if input_data is not None else Tensor([])
        if self._neural_network.gradient == 'auto':
            return self._neural_network.forward(input_)
        
        return TorchLayer._TorchNNFunction.apply(input_, self._neural_network)
