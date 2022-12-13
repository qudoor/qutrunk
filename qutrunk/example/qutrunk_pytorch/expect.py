from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, Barrier,Ry


def run_expect(backend=None):
    # Create quantum circuit
    qc = QCircuit(backend=backend)

    # Allocate quantum qubits
    qr = qc.allocate(1)

    # Apply quantum gates
    H * qr[0]
    Barrier * qr
    Ry(1.72) * qr[0]

    # measure
    Measure * qr[0]

    # Run quantum circuit with 100 times
    res = qc.run(shots=100)
    print(res.get_counts())


    return qc


if __name__ == "__main__":
    # Run locally
    circuit = run_expect()

    # Draw quantum circuit
    circuit.draw()
