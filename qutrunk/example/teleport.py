"""Quantum Teleportation."""
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, X, Z, Measure, CNOT, U3


def run_teleport():
    # allocate
    qc = QCircuit()
    qureg = qc.allocate(3)

    # Prepare an initial state
    U3(0.3, 0.2, 0.1) | qureg[0]

    # Prepare a Bell pair
    H * qureg[1]
    CNOT * (qureg[1], qureg[2])

    # Measure in the Bell basis
    CNOT * (qureg[0], qureg[1])
    H * qureg[0]
    Measure * qureg[0]
    Measure * qureg[1]

    # TODO: have some problem.
    # Apply a correction
    Z * qureg[2]
    X * qureg[2]
    Measure * qureg[2]

    # run circuit
    res = qc.run(shots=1024)

    print(res.get_counts())
    # return circuit
    return qc


if __name__ == "__main__":
    circuit = run_teleport()
    circuit.draw()
