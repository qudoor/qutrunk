"""Quantum Fourier Transform examples."""

import math

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import All, H, Measure, P
from qutrunk.circuit.ops import QFT


def full_qft():
    circuit = QCircuit()
    qreg = circuit.allocate(5)

    QFT * qreg

    circuit.draw(line_length=1000)
    state = circuit.get_all_state()
    print(state)

    All(Measure) * qreg

    res = circuit.run(shots=1000)
    print(res.get_counts())


def partial_qft():
    circuit = QCircuit()
    qreg = circuit.allocate(5)

    qubits = list(qreg)[::-2]
    QFT * qubits

    circuit.draw(line_length=1000)
    state = circuit.get_all_state()
    print(state)

    All(Measure) * qreg

    res = circuit.run(shots=1000)
    print(res.get_counts())


def qft_single_wave():
    num_qubits = 4
    circuit = QCircuit()
    qreg = circuit.allocate(num_qubits)
    All(H) * qreg
    P(math.pi / 4) * qreg[0]
    P(math.pi / 2) * qreg[1]
    P(math.pi) | qreg[2]
    circuit.qft()
    print(circuit.get_all_state())
    circuit.run()

    return circuit


if __name__ == "__main__":
    # full_qft()
    partial_qft()
