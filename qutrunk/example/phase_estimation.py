"""Phase Estimation Example: T-gate"""

from math import pi

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import NOT, Barrier, P, All, Measure
from qutrunk.circuit.ops import QPE


def run_qpe(backend=None):
    """Estimate T-gate phase."""
    # allocate
    qc = QCircuit(backend=backend)
    q1, q2 = qc.allocate([3, 1])

    # Prepare our eigenstate |psi>
    NOT * q2[0]
    Barrier * q1
    Barrier * q2

    # apply QPE
    QPE(P(pi/4)) * (q1, q2)

    # measure q1
    All(Measure) * q1

    # run circuit
    result = qc.run(shots=100)

    # calculate the value of theta
    value = result.get_values(q1)
    if value:
        theta = value[-1] / 2 ** len(q1)
        print("θ=", theta)

    return qc


if __name__ == "__main__":
    circuit = run_qpe()




