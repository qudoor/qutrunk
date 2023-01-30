"""Quantum super-dense encoding example."""

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, CNOT, Measure, X, Z, I


def run_super_dense_encoding(message, backend=None):
    """
    Result: The result will always be like [{"0b10": 1024}]
    """
    # Create quantum circuit
    qc = QCircuit(backend=backend)

    # Allocate quantum qubits
    qr = qc.allocate(2)

    # Prepared bell state
    H * qr[0]
    CNOT * (qr[0], qr[1])
    
    # Encoding message
    if message == '01':
        X * qr[0]
    elif message == '10':
        Z * qr[0]
    elif message == '11':
        X * qr[0]
        Z * qr[0]
    else:
        I * qr[0]

    # Decoding message
    CNOT * (qr[0], qr[1])
    H * qr[0]
    
    # measure
    Measure * qr[0]
    Measure * qr[1]

    # Print quantum circuit
    qc.print()

    # Run quantum circuit with 1024 times
    res = qc.run(shots=1024)
    print(res.get_counts())

    # Print quantum circuit exection information
    print(res.running_info())

    return qc


if __name__ == "__main__":
    # Can encode for 00, 01, 10, 11
    circuit = run_super_dense_encoding('10')

    # Draw quantum circuit
    circuit.draw()
