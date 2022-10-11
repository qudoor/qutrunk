"""Example of a simple quantum random number generator."""

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, All, X, CNOT


def deutsch(backend=None):
    # allocate
    qc = QCircuit(backend=backend)
    qureg = qc.allocate(2)

    X * qureg[1]
    All(H) * qureg
    CNOT * (qureg[0], qureg[1])
    H * qureg[0]
    All(Measure) * qureg

    qc.print()
    result = qc.run()
    print("The result of Deutsch: ", result.get_measure()[0])
    return qc


if __name__ == "__main__":
    # local run
    circuit = deutsch()
    circuit.draw()
