from math import pi

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import *


@def_gate
def my_gate(a, b):
    return Gate() << (H, a) << (CNOT, (a, b))


def run_gates():
    # allocate
    qc = QCircuit()
    qr = qc.allocate(3)

    # apply gate
    H * qr[0]
    CNOT * (qr[0], qr[1])
    NOT * qr[0]
    Toffoli * (qr[0], qr[1], qr[2])
    P(pi / 2) * qr[2]
    R(pi / 2, pi / 2) * qr[0]
    Rx(pi / 2) * qr[1]
    Ry(pi / 2) * qr[1]
    Rz(pi / 2) * qr[1]
    S * qr[0]
    Sdg * qr[0]
    T * qr[0]
    Tdg * qr[0]
    X * qr[2]
    Y * qr[2]
    Z * qr[2]
    X1 * qr[0]
    Y1 * qr[0]
    Z1 * qr[0]
    Swap * (qr[0], qr[1])
    iSwap(pi / 2) * (qr[0], qr[1])
    SqrtX * qr[0]

    CX * (qr[0], qr[1])
    CY * (qr[0], qr[1])
    CZ * (qr[0], qr[1])
    CP(pi / 2) * (qr[0], qr[1])
    CR(pi / 2) * (qr[0], qr[1])
    CRx(pi / 2) * (qr[0], qr[1])
    CRy(pi / 2) * (qr[0], qr[1])
    CRz(pi / 2) * (qr[0], qr[1])
    MCX(2) * (qr[0], qr[1], qr[2])
    MCZ(2) * (qr[0], qr[1], qr[2])

    Rxx(pi / 2) * (qr[0], qr[1])
    Ryy(pi / 2) * (qr[0], qr[1])
    Rzz(pi / 2) * (qr[0], qr[1])

    U1(pi / 2) * qr[0]
    U2(pi / 2, pi / 2) * qr[0]
    U3(pi / 2, pi / 2, pi / 2) * qr[0]
    CU(pi / 2, pi / 2, pi / 2, pi / 2) * (qr[0], qr[1])
    CU1(pi / 2) * (qr[1], qr[2])
    CU3(pi / 2, pi / 2, pi / 2) * (qr[0], qr[1])
    I * qr[0]

    CH * (qr[0], qr[1])
    CSwap * (qr[0], qr[1], qr[2])
    CSqrtX * (qr[0], qr[1])
    SqrtXdg * qr[0]

    Barrier * qr

    Power(2, H) * qr[0]
    my_gate * (qr[0], qr[1])
    Power(2, my_gate) * (qr[0], qr[1])

    # measure
    All(Measure) * qr

    # print circuit
    qc.print()
    # print qasm circuit
    qc.print_qasm()

    qc.run()

    return qc


if __name__ == "__main__":
    circuit = run_gates()
    circuit.draw(line_length=3000)
