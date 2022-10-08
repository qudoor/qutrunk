"""Addition operation."""

from qutrunk.circuit import Qureg
from qutrunk.circuit.ops.operator import Operator, OperatorContext
from qutrunk.circuit.gates import X, MCX


class ADD(Operator):
    """Self-increment operation"""

    def __init__(self, number):
        super().__init__()
        if number < 0:
            raise ValueError("number must be more than zero.")
        self.number = number

    def _add_statement(self, qr):
        qr[0].circuit.append_statement(f"ADD({self.number}) * q")

    def __mul__(self, qr: Qureg):
        if not isinstance(qr, Qureg):
            raise TypeError("The operand must be Qureg.")

        num_qubits = len(qr)

        with OperatorContext(qr[0].circuit):
            for _ in range(self.number):
                ctrl = []
                for i in range(num_qubits, 1, -1):
                    for j in range(i - 1):
                        ctrl.append(qr[j])
                    MCX(i - 1) * (*ctrl, qr[i - 1])
                    ctrl = []

                X * qr[0]

        self._add_statement(qr)
