from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import CNOT, Measure, Toffoli, X, All, MCX
from qutrunk.circuit.ops import QSP
from qutrunk.circuit.ops import ADD


def increment(num_qubits, init_value):
    circuit = QCircuit()
    qr = circuit.allocate(num_qubits)

    QSP(init_value) * qr
    ADD * qr

    All(Measure) * qr
    res = circuit.run()
    print(res.get_outcome())

    return circuit


def decrement(num_qubits, init_value):
    circuit = QCircuit()
    qr = circuit.allocate(num_qubits)

    QSP(init_value) * qr

    X * qr[0]
    ctrl = []
    for i in range(1, num_qubits+1, 1):
        for j in range(i - 1):
            ctrl.append(qr[j])
        if i > 1:
            MCX(i-1) * (*ctrl, qr[i-1])
        ctrl = []

    All(Measure) * qr
    res = circuit.run()
    print(res.get_outcome())

    return circuit


if __name__ == "__main__":
    circuit = decrement(4, 0)
    circuit.draw()
