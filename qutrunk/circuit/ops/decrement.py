"""Self-decrement operation."""

from qutrunk.circuit import Qureg
from qutrunk.circuit.ops.operator import Operator, OperatorContext
from qutrunk.circuit.gates import X, MCX


class Decrement(Operator):
    """Self-decrement operation"""

    def __init__(self):
        super().__init__()

    def _add_statement(self, qr):
        qr[0].circuit.append_statement("DEC * q")

    def __mul__(self, qr: Qureg):
        if not isinstance(qr, Qureg):
            raise TypeError("The operand must be Qureg.")

        num_qubits = len(qr)
        X * qr[0]

        with OperatorContext(qr[0].circuit):
            ctrl = []
            for i in range(1, num_qubits):
                for j in range(i):
                    ctrl.append(qr[j])
                MCX(i) * (*ctrl, qr[i])
                ctrl = []

        self._add_statement(qr)


DEC = Decrement()
