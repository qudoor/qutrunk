from qutrunk.circuit.gates import All, H
from qutrunk.circuit import Qureg
from qutrunk.circuit.ops import QSP
from qutrunk.exceptions import QuTrunkError

class Plus(QSP):
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
            print(circuit.get_all_state())
    """

    def __init__(self):
        super().__init__('+')

    def __str__(self):
        return "PLUS"

    def __mul__(self, qureg: Qureg):
        if qureg.circuit.gates_len > 0:
            raise QuTrunkError("PLUS should be applied at the beginning of circuit.")

        super().__mul__(qureg)

    def _check_state(self, qureg: Qureg):
        if self.state == "+":
            return True

        return False

    def _process_state(self, qureg: Qureg):
        """Process plus state."""
        All(H) * qureg

    def _append_statement(self, qureg: Qureg):
        qureg.circuit.append_statement("PLUS * q")


PLUS = Plus()