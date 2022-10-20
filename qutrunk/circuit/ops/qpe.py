"""Quantum Phase Estimation."""

from qutrunk.circuit.ops.operator import Operator, OperatorContext
from qutrunk.circuit.gates import H, Measure, All, Barrier
from qutrunk.circuit.ops import IQFT


class QPE(Operator):
    def __init__(self, unitary):
        """Initialize a QPE gate."""
        super().__init__()
        self.unitary = unitary

    def __str__(self):
        """Return a string representation of the object."""
        return f"QPE({str(self.unitary)})"

    def __mul__(self, qreg):
        qreg_first = qreg[0]
        qreg_second = qreg[1]

        with OperatorContext(qreg_first[0].circuit):
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
            All(Measure) * qreg_first

    # TODO:_add_statement

