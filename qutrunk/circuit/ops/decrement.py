"""Self-decrement operation."""

from qutrunk.circuit import Qureg
from qutrunk.circuit.ops.operator import Operator, OperatorContext
from qutrunk.circuit.gates import X, MCX


class Decrement(Operator):
    """Self-decrement operation.

    Example:
        .. code-block:: python

            from qutrunk.circuit import QCircuit
            from qutrunk.circuit.gates import Measure, All
            from qutrunk.circuit.ops import QSP
            from qutrunk.circuit.ops import DEC

            circuit = QCircuit()
            qr = circuit.allocate(4)
            QSP(0) * qr
            DEC * qr
            All(H) * qr
            res = circuit.run()
            print(res.get_outcome())

    """

    def __init__(self):
        super().__init__()

    def _add_statement(self, qr):
        qr[0].circuit.append_statement("DEC * q")

    def __mul__(self, qr: Qureg):
        if not isinstance(qr, Qureg):
            raise TypeError("The operand must be Qureg.")

        num_qubits = len(qr)

        with OperatorContext(qr[0].circuit):
            X * qr[0]
            ctrl = []
            for i in range(1, num_qubits):
                for j in range(i):
                    ctrl.append(qr[j])
                MCX(i) * (*ctrl, qr[i])
                ctrl = []

        self._add_statement(qr)


DEC = Decrement()
