import sys
from numpy import pi
from qutrunk.circuit.ops import PauliX, PauliY, PauliZ, PauliI, PauliArrary, PauliType, PauliCoeffi
from qutrunk.backends import BackendQuSprout, ExecType
from qutrunk.circuit.gates import H, Measure, All, CR
from qutrunk.circuit.circuit import QCircuit


def test_exp(backend=None):
    # allocate
    qibitnum = 2
    qc = QCircuit(backend=backend)
    qureg = qc.allocate(qibitnum)
    H * qureg[0]
    H * qureg[1]
    CR(pi/2) * (qureg[0], qureg[1])

    qc, qureg = qc.inverse()

    print("PauliI期望值：", qc.expval(PauliI(0)))
    print("PauliX期望值：", qc.expval(PauliX(0)))
    print("PauliY期望值：", qc.expval(PauliY(0)))
    print("PauliZ期望值：", qc.expval(PauliZ(0)))
    pauillist = PauliArrary([PauliI(0), PauliX(1)])
    print("PauliIX期望值：", qc.expval(pauillist))

    print("乘积之和的期望值：", qc.expval_sum(PauliCoeffi([PauliType.POT_PAULI_X, PauliType.POT_PAULI_X, PauliType.POT_PAULI_I, PauliType.POT_PAULI_I], [99.0, 77.0]), qibitnum))

    All(Measure) * qureg

    result = qc.run()

    print("执行CR，测量得到的结果是：", result.get_measure())
    return qc


if __name__ == "__main__":
    # local run
    circuit = test_exp()

    # qusprout run
    # circuit =test_exp(backend=BackendQuSprout())

    # qusprout mpi run
    # test_exp(backend=BackendQuSprout(ExecType.Mpi))
    print(circuit.draw())