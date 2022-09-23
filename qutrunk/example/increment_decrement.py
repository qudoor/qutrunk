from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import CNOT, Measure, Toffoli, X, All, MCX
from qutrunk.circuit.ops import QSP


def increment(num_qubits, initvalue):
    circuit = QCircuit()
    qr = circuit.allocate(num_qubits)

    QSP(initvalue) * qr

    MCX(3) * (qr[0], qr[1], qr[2], qr[3])
    Toffoli * (qr[0], qr[1], qr[2])
    CNOT * (qr[0], qr[1])
    X * qr[0]

    All(Measure) * qr
    res = circuit.run()
    print(res.get_outcome())

    return circuit


def decrement(num_qubits, initvalue):
    circuit = QCircuit()
    qr = circuit.allocate(num_qubits)

    QSP(initvalue) * qr

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
