"""Quantum Fourier Transform examples."""

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import All, Measure
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


if __name__ == "__main__":
    full_qft()
