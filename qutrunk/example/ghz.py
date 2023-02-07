"""GHZ state example."""

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import CX, Measure, H, Barrier, All


def run_ghz(backend=None):
    """
    Result: The result will always be like [{"0b000": xxx}, {"0b111": xxx}]
    """
    # Create quantum circuit
    qc = QCircuit(name="ghz", backend=backend)

    # Allocate quantum qubits
    qr = qc.allocate(3)

    # Create a GHZ state
    H * qr[0]
    CX * (qr[0], qr[1])
    CX * (qr[0], qr[2])

    Barrier * qr

    # Measure all the qubits
    All(Measure) * qr

    # Run quantum circuit with 1024 times
    res = qc.run(shots=1024)

    # Print measure results like:
    # [{"000": 527}, {"111": 497}]
    print(res.get_counts())
    return qc


if __name__ == "__main__":
    # Run locally
    circuit = run_ghz()

    # Dram quantum circuit
    circuit.draw()

