import numpy as np
import json

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
        self.theta = self._circuit.create_parameters(['theta', ""])

        # TODO: to list
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
        # thetas是列表， [0.104]
        # list to dict
        t = [{"theta": theta} for theta in thetas]
        print("t=", t)
        self._circuit.bind_parameters(t[0])
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


if __name__ == '__main__':
    circuit = QuantumCircuit(1, None, 100)
    # circuit.run({"theta": np.pi})  circuit.run([np.pi])[0]
    # print(f'Expected value for rotation pi= {circuit.run({"theta": np.pi})[0]}')
    print(f'Expected value for rotation pi= {circuit.run([np.pi])[0]}')
    circuit._circuit.draw()
