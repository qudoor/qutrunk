"""Quantum Teleportation."""

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, X, Z, Measure, CNOT, U3


def run_teleport():
    # Create quantum circuit
    qc = QCircuit()

    # Allocate 3 quantum qubits
    qureg = qc.allocate(3)

    # Apply U3 gate on the first quantum qubit
    U3(0.3, 0.2, 0.1) | qureg[0]

    # Prepare a Bell pair
    H * qureg[1]
    CNOT * (qureg[1], qureg[2])
    CNOT * (qureg[0], qureg[1])
    H * qureg[0]

    # Measure in the Bell basis
    Measure * qureg[0]
    Measure * qureg[1]

    # TODO: have some problem.
    # Apply a correction
    Z * qureg[2]
    X * qureg[2]

    # Measure in the Bell basis
    Measure * qureg[2]

    # Run circuit
    res = qc.run(shots=1024)

    # Print execution results like:
    # [{"000": 3}, {"001": 273}, {"010": 260}, {"011": 6}, {"100": 6}, {"101": 234}, {"110": 232}, {"111": 10}]
    print(res.get_counts())

    return qc


if __name__ == "__main__":
    # Run locally
    circuit = run_teleport()

    # Dram quantum circuit
    circuit.draw()
