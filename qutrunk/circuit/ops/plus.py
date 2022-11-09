from qutrunk.circuit.gates import All, H
from qutrunk.circuit import Qureg
from qutrunk.circuit.ops import Operator
from qutrunk.exceptions import QuTrunkError


class Plus(Operator):
    """Quantum state preparation Operator.

    Init the quantum state to plus state.

    Example:
        .. code-block:: python

            from qutrunk.circuit.ops import PLUS
            from qutrunk.circuit import QCircuit
            from qutrunk.circuit.gates import H, All, Measure

            circuit = QCircuit()
            qureg = circuit.allocate(2)
            PLUS * qureg
            print(circuit.get_statevector())
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "PLUS"

    def __mul__(self, qureg: Qureg):
        if qureg.circuit.num_gates > 0:
            raise QuTrunkError("PLUS should be applied at the beginning of circuit.")

        if not isinstance(qureg, Qureg):
            raise TypeError("the operand must be Qureg.")

        self._process_state(qureg)

    def _process_state(self, qureg: Qureg):
        """Process plus state."""
        All(H) * qureg


PLUS = Plus()
