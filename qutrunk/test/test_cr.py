import sys

sys.path.append("D:/code/qike/qubox/src")
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, All, CR
from qutrunk.backends import BackendQuSprout, ExecType
from numpy import pi


def testcr(backend=None):
    # allocate
    qc = QCircuit(backend=backend)
    qureg = qc.allocate(2)
    H * qureg[0]
    H * qureg[1]
    CR(pi / 2) * (qureg[0], qureg[1])

    qc, qureg = qc.inverse()

    All(Measure) * qureg
    qc.print()
    result = qc.run()

    print("执行CR，测量得到的结果是：", result.get_measure()[0])
    return qc


if __name__ == "__main__":
    qc = testcr()
    print(qc.draw())
