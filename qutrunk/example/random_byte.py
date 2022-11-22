"""Quantum random number generator."""

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, All


def run_random_byte(backend=None):
    # Create circuit with local simulator
    qc = QCircuit(backend)

    # Allocate 8 quantuam qubits
    qureg = qc.allocate(8)

    # Apply quantum gates
    All(H) * qureg

    # Measure all quantum qubits
    All(Measure) * qureg

    # Print circuit
    qc.print()

    # Run circuit
    res = qc.run()

    # Print measure result like:
    # [0, 1, 0, 0, 0, 0, 1, 0]
    meas = res.get_measures()
    reslen = len(meas)
    if reslen > 0:
        print(meas[reslen-1])

    return qc


if __name__ == "__main__":
    # Run locally
    circuit = run_random_byte()
