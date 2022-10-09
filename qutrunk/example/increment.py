"""Self-increment operation  example."""
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Measure, All
from qutrunk.circuit.ops import Classical
from qutrunk.circuit.ops import INC


def increment(num_qubits, init_value):
    circuit = QCircuit()
    qr = circuit.allocate(num_qubits)

    Classical(init_value) * qr

    INC * qr

    All(Measure) * qr
    res = circuit.run()
    print(res.get_outcome())

    return circuit


if __name__ == "__main__":
    circuit = increment(4, 0)
    # circuit.print(unroll=False)

