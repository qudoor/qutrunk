"""addition example."""
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Measure, All
from qutrunk.circuit.ops import QSP
from qutrunk.circuit.ops import ADD


def run_addition(num_qubits, init_value, number=0):
    circuit = QCircuit()
    qr = circuit.allocate(num_qubits)

    QSP(init_value) * qr

    ADD(number) * qr

    All(Measure) * qr
    res = circuit.run()
    print(res.get_outcome())

    return circuit


if __name__ == "__main__":
    circuit = run_addition(4, 0, 5)
