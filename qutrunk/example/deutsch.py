"""Example of a simple quantum random number generator."""

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, All, X, CNOT


def deutsch(backend=None):
    # Create quantum circuit
    qc = QCircuit(backend=backend)

    # Allocate quantum qubits
    qureg = qc.allocate(2)

    # Apply quantum gates
    X * qureg[1]
    All(H) * qureg
    CNOT * (qureg[0], qureg[1])
    H * qureg[0]

    # Measure all quantum qubits
    All(Measure) * qureg

    # Print quantum circuit
    qc.print()

    # Run quantum circuit and print result
    result = qc.run()
    print("The result of Deutsch: ", result.get_measure()[0])

    return qc


if __name__ == "__main__":
    # Run locally
    circuit = deutsch()

    # Dram quantum circuit
    circuit.draw()
