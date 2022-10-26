from typing import Union

from qutrunk.circuit import Qureg
from qutrunk.circuit.ops import QSP
from qutrunk.exceptions import QuTrunkError
from qutrunk.circuit.gates import X


class Classical(QSP):
    """Quantum state preparation Operator.

    Init the quantum state to classical state.

    Args:
        state: The target state of circuit initialized to.

    Example:
        .. code-block:: python

            from qutrunk.circuit.ops import Classical
            from qutrunk.circuit import QCircuit
            from qutrunk.circuit.gates import H, All, Measure

            circuit = QCircuit()
            qureg = circuit.allocate(2)
            Classical("0100") * qureg | Classical(4) * qureg
            print(circuit.get_all_state())
    """

    def __init__(self, state: Union[str, int]):
        super().__init__(state)

    def __str__(self):
        return "Classical"

    def __mul__(self, qureg: Qureg):
        if qureg.circuit.num_gates > 0:
            raise QuTrunkError("Classical should be applied at the beginning of circuit.")

        super().__mul__(qureg)

    def _check_state(self, qureg: Qureg):
        if isinstance(self.state, str):
            val = int(self.state, base=2)
            if 0 <= val < 2 ** len(qureg):
                return True

        if isinstance(self.state, int):
            if 0 <= self.state < 2 ** len(qureg):
                return True

        return False

    def _process_state(self, qureg: Qureg):
        bit_strs = ""
        if isinstance(self.state, str):
            if self.state.startswith("0b"):
                bit_strs = self.state[2:]
            else:
                bit_strs = self.state
        elif isinstance(self.state, int):
            bit_strs = str(bin(self.state))
            bit_strs = bit_strs[2:]

        bit_strs = bit_strs.zfill(len(qureg))
        bit_strs = bit_strs[::-1]

        for i, _ in enumerate(qureg):
            if bit_strs[i] == "1":
                X * qureg[i]

    def _append_statement(self, qureg: Qureg):
        qureg.circuit.append_statement(f"Classical('{self.state}') * q")