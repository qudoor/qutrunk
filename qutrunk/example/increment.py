"""self increment example."""
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import CNOT, Measure, Toffoli, X, All, MCX
from qutrunk.circuit.ops import QSP


def increment(num_qubits, init_value):
    # Create quantum circuit 
    circuit = QCircuit()

    # Allocate quantum qubits
    qr = circuit.allocate(num_qubits)

    # Set initial amplitudes to classical state with init_value
    QSP(init_value) * qr

    # Apply quantam gates
    MCX(3) * (qr[0], qr[1], qr[2], qr[3])
    Toffoli * (qr[0], qr[1], qr[2])
    CNOT * (qr[0], qr[1])
    X * qr[0]

    # Measure all quantum qubits
    All(Measure) * qr

    # Run quantum circuit
    res = circuit.run()

    # Print measure result like:
    # 0b0001
    print(res.get_outcome())

    return circuit


if __name__ == "__main__":
    # Run locally
    circuit = increment(4, 0)

    # Dram quantum circuit
    circuit.draw()
