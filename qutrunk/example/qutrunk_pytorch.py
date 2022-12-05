import numpy as np
import matplotlib.pyplot as plt
import json

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import qutrunk
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Ry, Measure, Barrier


class QuantumCircuit:

    def __init__(self, n_qubits, backend=None, shots=100):
        # 定义量子电路
        # Create quantum circuit
        self._circuit = QCircuit(backend)
        # Allocate quantum qubits
        qr = self._circuit.allocate(n_qubits)

        # 参数化 theta
        self.theta = self._circuit.create_parameters(['theta'])

        # 应用量子门
        H * qr[0]
        Barrier * qr
        Ry(self.theta[0]) * qr[0]

        # 测量
        Measure * qr[0]

        # 指定运行量子计算后端
        # self.backend = backend
        # 运行次数
        self.shots = shots

    def run(self, thetas):
        self._circuit.bind_parameters(thetas)
        result = self._circuit.run(shots=self.shots)
        result = result.get_counts()  # <class 'str'> [{"0": 57}, {"1": 43}]
        result = json.loads(result)  # <class 'list'> [{'0': 57}, {'1': 43}]

        counts = []
        states = []
        for r in result:
            for key, value in r.items():
                states.append(key)
                counts.append(value)

        states = np.array(states).astype(float)  # [0. 1.]
        counts = np.array(counts)  # [51 49]

        # Compute probabilities for each state
        probabilities = counts / self.shots  # [0.51 0.49]
        # Get state expectation
        expectation = np.sum(states * probabilities)

        # print(np.array([expectation]))  # [0.49]
        return np.array([expectation])


class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit
        # TODO:modify
        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())

        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift

        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left = ctx.quantum_circuit.run(shift_left[i])

            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None


class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """

    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(1, backend, shots)
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)


if __name__ == '__main__':
    circuit = QuantumCircuit(1, None, 100)
    # circuit.run({"theta": np.pi})
    print(f'Expected value for rotation pi= {circuit.run({"theta": np.pi})[0]}')
    circuit._circuit.draw()
