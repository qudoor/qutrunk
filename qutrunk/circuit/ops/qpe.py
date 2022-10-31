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

            def _bin_int(itrable):
                return int("".join(map(str, reversed(itrable))), base=2)

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

            # print circuit
            # qc.print()

            # run circuit
            qc.run(shots=100)

            # print result
            print(q1.to_cl())

            # calculate the value of theta
            f = _bin_int(q1.to_cl())
            theta = f / 2 ** len(q1)
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
                self.unitary.ctrl() * (qreg_first[counting_qubit], qreg_second[0])
            repetitions *= 2
        # 3 inverse QFT
        Barrier * qreg_first
        Barrier * qreg_second
        IQFT * qreg_first

        # 4 measure qreg_first
        # All(Measure) * qreg_first


