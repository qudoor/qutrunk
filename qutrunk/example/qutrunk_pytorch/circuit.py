import numpy as np
import json

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Ry, Measure, Barrier


class QuantumCircuit:

    def __init__(self, shots=100):
        self.shots = shots

    def _new_circuit(self, theta):
        # Create quantum circuit
        self._circuit = QCircuit()
        # Allocate quantum qubits
        qr = self._circuit.allocate(1)
        # apply gates
        H * qr[0]
        Barrier * qr
        Ry(theta) * qr[0]
        # measure
        Measure * qr[0]

    def run(self, theta):
        self._new_circuit(theta)

        result = self._circuit.run(shots=self.shots)
        result = result.get_counts()
        result = json.loads(result)

        counts = []
        states = []
        for r in result:
            for key, value in r.items():
                states.append(int(key, base=2))
                counts.append(value)

        states = np.array(states).astype(float)
        counts = np.array(counts)

        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)

        return np.array([expectation])


if __name__ == '__main__':
    circuit = QuantumCircuit()
    print(f'Expected value for rotation pi= {circuit.run(np.pi)[0]}')
    circuit._circuit.draw()
