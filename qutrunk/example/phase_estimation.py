"""Phase Estimation Example: T-gate"""

from math import pi

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, NOT, Measure, CP, All
from qutrunk.circuit.ops import IQFT


def _bin_int(itrable):
    return int("".join(map(str, reversed(itrable))), base=2)


def run_phase_estimation(backend=None):
    """Estimate T-gate phase."""
    # allocate
    qc = QCircuit(backend=backend)
    q1, q2 = qc.allocate([3, 1])

    # stage 1
    NOT * q2[0]
    for qubit in range(3):
        H * q1[qubit]

    repetitions = 1
    for counting_qubit in range(3):
        for i in range(repetitions):
            CP(pi / 4) * (q1[counting_qubit], q2[0])
        repetitions *= 2

    # stage 2
    IQFT * q1

    # stage 3
    All(Measure) * q1

    # print circuit
    qc.print()

    # # run circuit
    qc.run(shots=100)

    # print result
    print(q1.to_cl())

    # stage 4: calculate the value of theta
    f = _bin_int(q1.to_cl())
    theta = f / 2 ** len(q1)
    print(theta)

    return qc


if __name__ == "__main__":
    circuit = run_phase_estimation()
    circuit.draw()
