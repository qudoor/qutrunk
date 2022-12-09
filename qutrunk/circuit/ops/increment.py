"""Self-increment operation."""

from qutrunk.circuit import Qureg
from qutrunk.circuit.ops.operator import Operator
from qutrunk.circuit.gates import X, MCX


class Increment(Operator):
    """Self-increment operation.

    Example:
        .. code-block:: python

            from qutrunk.circuit import QCircuit
            from qutrunk.circuit.gates import Measure, All
            from qutrunk.circuit.ops import Classical
            from qutrunk.circuit.ops import INC

            circuit = QCircuit()
            qr = circuit.allocate(4)
            Classical(0) * qr
            INC * qr
            All(Measure) * qr
            res = circuit.run()
            print(res.get_bitstrs())

    """

    def __init__(self):
        super().__init__()

    def __mul__(self, qr: Qureg):
        if not isinstance(qr, Qureg):
            raise TypeError("The operand must be Qureg.")

        num_qubits = len(qr)

        ctrl = []
        for i in range(num_qubits, 1, -1):
            for j in range(i - 1):
                ctrl.append(qr[j])
            MCX(i - 1) * (*ctrl, qr[i - 1])
            ctrl = []

        X * qr[0]


INC = Increment()
