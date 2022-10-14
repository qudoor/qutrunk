"""GHZ state example."""

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import CX, Measure, H, Barrier, All


def run_ghz(qubits=5, backend=None):
    # Create quantum circuit
    qc = QCircuit(name="ghz", backend=backend)

    # Allocate quantum qubits
    qr = qc.allocate(qubits)

    # Create a GHZ state
    H * qr[0]
    for i in range(qubits - 1):
        CX * (qr[i], qr[i + 1])
    Barrier * qr

    # Measure all of the qubits
    All(Measure) * qr

    # Run quantum circuit with 1024 times
    res = qc.run(shots=1024)

    # Print measure results like:
    # [{"00000": 536}, {"11111": 488}]
    print(res.get_counts())
    return qc


if __name__ == "__main__":
    # Run locally
    circuit = run_ghz()

    # Dram quantum circuit
    circuit.draw()

