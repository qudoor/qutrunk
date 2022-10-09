"""Quantum random number generator."""

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, All


def run_random_byte(backend=None):
    # allocate
    qc = QCircuit(backend)

    qureg = qc.allocate(8)

    All(H) * qureg
    All(Measure) * qureg

    # print circuit
    qc.print()

    # run circuit
    res = qc.run()

    print(res.get_measure())
    return qc


if __name__ == "__main__":
    # local run
    circuit = run_random_byte()
    circuit.draw()
