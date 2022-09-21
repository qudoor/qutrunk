import sys

sys.path.append("D:/code/qike/qubox/src")
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, All, CU, CU1, CU3
from qutrunk.backends import BackendQuSprout, ExecType
from numpy import pi


def testcu(backend=None):
    # allocate
    qc = QCircuit(backend=backend)
    qureg = qc.allocate(2)
    H | qureg[0]
    H | qureg[1]
    CU(pi, pi / 2, pi / 3, pi / 4) | (qureg[0], qureg[1])

    # qc, qureg = qc.inverse()

    All(Measure) | qureg
    qc.print()
    result = qc.run()

    print("执行CU，测量得到的结果是：", result.get_measure()[0])
    return qc


def testcu1(backend=None):
    # allocate
    qc = QCircuit(backend=backend)
    qureg = qc.allocate(2)
    H | qureg[0]
    H | qureg[1]
    CU1(pi / 2) | (qureg[0], qureg[1])

    # qc, qureg = qc.inverse()

    All(Measure) | qureg
    qc.print()
    result = qc.run()

    print("执行CU1，测量得到的结果是：", result.get_measure()[0])
    return qc


def testcu3(backend=None):
    # allocate
    qc = QCircuit(backend=backend)
    qureg = qc.allocate(2)
    H | qureg[0]
    H | qureg[1]
    CU3(pi, pi / 2, pi / 3) | (qureg[0], qureg[1])

    qc, qureg = qc.inverse()

    All(Measure) | qureg
    qc.print()
    result = qc.run()

    print("执行CU3，测量得到的结果是：", result.get_measure()[0])
    return qc


if __name__ == "__main__":
    qc = testcu()
    qc.draw()

    qcy = testcu1()
    qcy.draw()

    qcz = testcu3()
    qcz.draw()
