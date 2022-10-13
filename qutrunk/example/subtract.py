"""subtract example."""
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Measure, All
from qutrunk.circuit.ops import Classical
from qutrunk.circuit.ops import Subtract


def run_subtract(num_qubits, init_value, number=0):
    circuit = QCircuit()
    qr = circuit.allocate(num_qubits)

    Classical(init_value) * qr

    Subtract(number) * qr

    All(Measure) * qr
    res = circuit.run()
    print(res.get_outcome())

    return circuit


if __name__ == "__main__":
    circuit = run_subtract(4, 3, 3)

