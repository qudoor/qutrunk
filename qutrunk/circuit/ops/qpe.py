"""Quantum Phase Estimation."""

from qutrunk.circuit.ops.operator import Operator
from qutrunk.circuit.gates import H, All, Barrier
from qutrunk.circuit.ops import IQFT


class QPE(Operator):
    """Quantum Phase Estimation.

    Args:
        unitary: Stand Gate.

    Example:
        .. code-block:: python

            from math import pi
            from qutrunk.circuit import QCircuit
            from qutrunk.circuit.gates import NOT, Barrier, P, All, Measure
            from qutrunk.circuit.ops import QPE

            # allocate
            qc = QCircuit()
            q1, q2 = qc.allocate([3, 1])

            # Prepare our eigenstate |psi>
            NOT * q2[0]
            Barrier * q1
            Barrier * q2

            # apply QPE
            QPE(P(pi/4)) * (q1, q2)

            # measure q1
            All(Measure) * q1

            # print circuit
            # qc.print()

            # run circuit
            result = qc.run()

            # calculate the value of theta
            vals = result.get_values(q1)
            theta = vals[0] / 2 ** len(q1)
            print("θ=", theta)

    """
    def __init__(self, unitary):
        """Initialize a QPE gate."""
        super().__init__()
        # TODO: whether to Gate.
        self.unitary = unitary

    def __str__(self):
        """Return a string representation of the object."""
        return f"QPE({str(self.unitary)})"

    def __mul__(self, qreg):
        qreg_first = qreg[0]
        qreg_second = qreg[1]

        # 1 apply H gate
        All(H) * qreg_first

        # 2 apply cu
        repetitions = 1
        num_qubits = len(qreg_first)
        for counting_qubit in range(num_qubits):
            for i in range(repetitions):
                # apply cu gate
                self.unitary.ctrl() * (qreg_first[counting_qubit], *qreg_second)
            repetitions *= 2
        # 3 inverse QFT
        Barrier * qreg_first
        Barrier * qreg_second
        IQFT * qreg_first



