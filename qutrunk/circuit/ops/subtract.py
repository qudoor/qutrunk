"""Subtract operation."""

from qutrunk.circuit import Qureg
from qutrunk.circuit.ops.operator import Operator, OperatorContext
from qutrunk.circuit.gates import X, MCX


class Subtract(Operator):
    """Subtract operation.

    Example:
        .. code-block:: python

            from qutrunk.circuit import QCircuit
            from qutrunk.circuit.gates import Measure, All
            from qutrunk.circuit.ops import QSP
            from qutrunk.circuit.ops import ADD

            circuit = QCircuit()
            qr = circuit.allocate(4)
            QSP(3) * qr
            Subtract(3) * qr
            All(H) * qr
            res = circuit.run()
            print(res.get_outcome())
    """

    def __init__(self, number):
        super().__init__()
        if number < 0:
            raise ValueError("number must be more than zero.")
        self.number = number

    def _add_statement(self, qr):
        qr[0].circuit.append_statement(f"Subtract({self.number}) * q")

    def __mul__(self, qr: Qureg):
        if not isinstance(qr, Qureg):
            raise TypeError("The operand must be Qureg.")

        num_qubits = len(qr)

        with OperatorContext(qr[0].circuit):
            for _ in range(self.number):
                X * qr[0]
                ctrl = []
                for i in range(1, num_qubits):
                    for j in range(i):
                        ctrl.append(qr[j])
                    MCX(i) * (*ctrl, qr[i])
                    ctrl = []

        self._add_statement(qr)
