import logging
from numbers import Integral
from typing import Tuple, Union, List, Callable, Optional, cast, Iterable
from enum import Enum
import copy

import numpy as np
from qutrunk.qml.neural_network import NeuralNetwork
#from .quantum_circuit import QuantumCircuit
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import All, H, Ry, Barrier, Measure
from qutrunk.qml.interface import MLInterface 
from qutrunk.qml.gradient.parameter_shift import parameter_shift


# note: CircuitQNN不绑定具体机器学习框架数据类型，通过外面传入interface参数确定
class CircuitQNN(NeuralNetwork):
    def __init__(self, circuit_func, interface='pytorch', grad='auto'):
        self.circuit_func = circuit_func
        self.grad = grad
        self.origin_interface = interface
        self._interface = MLInterface(interface)

    @property
    def gradient(self):
        return self.grad

    @property
    def interface(self):
        return self._interface

    def _forward(self, input_data):
        np_value = self.circuit_func(input_data)
        if self.grad == 'auto' and self.origin_interface == 'pytorch':
            result = self._interface.tensor([np_value], requires_grad=True)
        else:
            result = self._interface.tensor([np_value])
        return result

    def _backward(self, input_data):
        """ Backward pass computation """
        if self.grad == 'parameter-shift':
            gradients = parameter_shift(self.circuit_func, input_data)
            return self._interface.tensor([gradients])