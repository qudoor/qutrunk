"""Self-decrement operation."""

from qutrunk.circuit import Qureg
from qutrunk.circuit.ops.operator import Operator
from qutrunk.circuit.gates import X, MCX


class Decrement(Operator):
    """Self-decrement operation.

    Example:
        .. code-block:: python

            from qutrunk.circuit import QCircuit
            from qutrunk.circuit.gates import Measure, All
            from qutrunk.circuit.ops import Classical
            from qutrunk.circuit.ops import DEC

            circuit = QCircuit()
            qr = circuit.allocate(4)
            Classical(4) * qr
            DEC * qr
            All(Measure) * qr
            res = circuit.run()
            print(res.get_outcome())

    """

    def __init__(self):
        super().__init__()

    def __mul__(self, qr: Qureg):
        if not isinstance(qr, Qureg):
            raise TypeError("The operand must be Qureg.")

        num_qubits = len(qr)

        X * qr[0]
        ctrl = []
        for i in range(1, num_qubits):
            for j in range(i):
                ctrl.append(qr[j])
            MCX(i) * (*ctrl, qr[i])
            ctrl = []


DEC = Decrement()
