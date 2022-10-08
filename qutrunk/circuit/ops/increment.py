"""Self-increment operation."""

from qutrunk.circuit import Qureg
from qutrunk.circuit.ops.operator import Operator, OperatorContext
from qutrunk.circuit.gates import X, MCX


class Increment(Operator):
    """Self-increment operation"""

    def __init__(self):
        super().__init__()

    def _add_statement(self, qr):
        qr[0].circuit.append_statement("INC * q")

    def __mul__(self, qr: Qureg):
        if not isinstance(qr, Qureg):
            raise TypeError("The operand must be Qureg.")

        num_qubits = len(qr)

        with OperatorContext(qr[0].circuit):
            ctrl = []
            for i in range(num_qubits, 1, -1):
                for j in range(i - 1):
                    ctrl.append(qr[j])
                MCX(i - 1) * (*ctrl, qr[i - 1])
                ctrl = []

            X * qr[0]

        self._add_statement(qr)


INC = Increment()
