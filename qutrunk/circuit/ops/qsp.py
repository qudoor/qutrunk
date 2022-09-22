"""Quantum state preparation Operator."""

import cmath
from typing import Union

from qutrunk.exceptions import QuTrunkError
from .operator import Operator, OperatorContext
from qutrunk.circuit import Qureg
from qutrunk.circuit.gates import X, All, H


class QSP(Operator):
    """Quantum state preparation Operator.

    Init the quantum state to specific state, currently support: Plus state, Classical state, amplitude state.

    Args:
        state: The target state of circuit initialized to.

    Example:
        .. code-block:: python

            from qutrunk.circuit.ops import QSP
            from qutrunk.circuit import QCircuit
            from qutrunk.circuit.gates import H, All, Measure

            circuit = QCircuit()
            qureg = circuit.allocate(2)
            QSP("+") * qureg
            print(circuit.get_all_state())
    """

    def __init__(self, state: Union[str, int], classicvector: list = None):
        super().__init__()
        self.state = state
        self.classicvector = classicvector

    def _check_state(self, qureg: Qureg):
        if self.state == "+":
            return True

        if self.state == "AMP":
            if 0 <= len(self.classicvector) <= 2 ** len(qureg):
                return True

        if isinstance(self.state, str):
            val = int(self.state, base=2)
            if 0 <= val < 2 ** len(qureg):
                return True

        if isinstance(self.state, int):
            if 0 <= self.state < 2 ** len(qureg):
                return True

        return False

    def __mul__(self, qureg: Qureg):
        """Apply the QSP operator."""
        if not isinstance(qureg, Qureg):
            raise TypeError("the operand must be Qureg.")

        if not self._check_state(qureg):
            raise ValueError("Invalid Classical state.")

        if qureg.circuit.gates_len > 0:
            raise QuTrunkError("QSP should be applied at the beginning of circuit.")

        with OperatorContext(qureg.circuit) as oc:
            if self.state == "+":
                self._process_plus_state(qureg)
            elif self.state == "AMP":
                self._process_amp_state(qureg)
            else:
                self._process_classical_state(qureg)

        if self.state == "AMP":
            qureg.circuit.append_statement(f"QSP('{self.state}', '{self.classicvector}') * q")
        else:
            qureg.circuit.append_statement(f"QSP('{self.state}') * q")

    def _process_amp_state(self, qureg: Qureg):
        """Process amp state."""
        if (len(self.classicvector) <= 0):
            return

        listsum = sum(self.classicvector)
        for element in self.classicvector:
            normalized_element = cmath.sqrt(complex(element / listsum))
            qureg.circuit.init_amp_reals.append(normalized_element.real)
            qureg.circuit.init_amp_imags.append(normalized_element.imag)

    def _process_plus_state(self, qureg: Qureg):
        """Process plus state."""
        All(H) * qureg

    def _process_classical_state(self, qureg: Qureg):
        """Process classical state."""
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