"""Phase Kickback example."""

from math import pi

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, NOT, Measure, CP


def run_phase_kickback(backend=None):
    # allocate
    qc = QCircuit(backend=backend)
    q1, q2 = qc.allocate([2, 1])

    # apply gate
    H * q1[0]
    H * q1[1]
    NOT * q2[0]
    CP(pi / 4) * (q1[0], q2[0])
    CP(pi / 2) * (q1[1], q2[0])
    Measure * q1[0]
    Measure * q1[1]
    Measure * q2[0]

    # print circuit
    qc.print()

    # run circuit
    res = qc.run()

    # print result
    print(res.get_measures())

    return qc


if __name__ == "__main__":
    circuit = run_phase_kickback()
    circuit.draw()

