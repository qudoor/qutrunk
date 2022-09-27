from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, All, X1, Y1, Z1


def test_x1(backend=None):
    # allocate
    qc = QCircuit(backend=backend)
    qureg = qc.allocate(1)
    H | qureg[0]
    X1 | qureg[0]

    # qc, qureg = qc.inverse()

    All(Measure) | qureg
    qc.print()
    result = qc.run()

    print("执行X1，测量得到的结果是：", result.get_measure()[0])
    return qc


def test_y1(backend=None):
    # allocate
    qc = QCircuit(backend=backend)
    qureg = qc.allocate(1)
    H | qureg[0]
    Y1 | qureg[0]

    qc, qureg = qc.inverse()

    All(Measure) | qureg
    qc.print()
    result = qc.run()

    print("执行Y1，测量得到的结果是：", result.get_measure()[0])
    return qc


def test_z1(backend=None):
    # allocate
    qc = QCircuit(backend=backend)
    qureg = qc.allocate(1)
    H | qureg[0]
    Z1 | qureg[0]

    qc, qureg = qc.inverse()

    All(Measure) | qureg
    qc.print()
    result = qc.run()

    print("执行Z1，测量得到的结果是：", result.get_measure()[0])
    return qc

