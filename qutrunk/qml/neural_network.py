from abc import ABC, abstractmethod
from typing import Tuple, Union, List, Optional

import numpy as np

class NeuralNetwork(ABC):
    def forward(self, input_data):
        output_data = self._forward(input_data)
        return output_data

    @abstractmethod
    def _forward(self, input_data):
        raise NotImplementedError

    def backward(self, input_data):
        input_grad = self._backward(input_data)
        return input_grad

    @abstractmethod
    def _backward(self, input_data):
        raise NotImplementedError
