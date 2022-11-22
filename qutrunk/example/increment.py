"""Self-increment operation  example."""

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Measure, All
from qutrunk.circuit.ops import Classical
from qutrunk.circuit.ops import INC


def increment(num_qubits, init_value):
    # Create quantum circuit
    circuit = QCircuit()

    # Allocate quantum qubits
    qr = circuit.allocate(num_qubits)

    # Set initial amplitudes to classical state with init_value
    Classical(init_value) * qr

    # Apply quantum gates
    INC * qr

    # Measure all quantum qubits
    All(Measure) * qr

    # Run quantum circuit
    res = circuit.run()

    # Print measure result like:
    # 0b0001
    print(res.get_bitstrs())

    return circuit


if __name__ == "__main__":
    # Run locally
    circuit = increment(4, 0)

    # Dram quantum circuit
    circuit.draw()
