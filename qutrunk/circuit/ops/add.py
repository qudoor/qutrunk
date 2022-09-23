"""Self-increment operation."""

from qutrunk.circuit import Qureg
from qutrunk.circuit.ops.operator import Operator, OperatorContext
from qutrunk.circuit.gates import X, MCX


class ADDOps(Operator):
    """Self-increment operation"""

    def __init__(self):
        super().__init__()

    def __mul__(self, qr: Qureg):
        print(type(qr))
        if not isinstance(qr, Qureg):
            raise TypeError("The operand must be Qureg.")

        num_qubits = len(qr)

        with OperatorContext(qr[0].circuit) as oc:
            ctrl = []
            for i in range(num_qubits, 1, -1):
                for j in range(i - 1):
                    ctrl.append(qr[j])
                MCX(i - 1) * (*ctrl, qr[i - 1])
                ctrl = []

            X * qr[0]


ADD = ADDOps()
