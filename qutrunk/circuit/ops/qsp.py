"""Quantum state preparation Operator."""
from abc import ABCMeta, abstractmethod
from typing import Union

from qutrunk.circuit.ops.operator import Operator
from qutrunk.circuit import Qureg


class QSP(Operator):
    """Quantum state preparation Operator.

    Parent class for CLASSICAL, PLUS, AMP

    Args:
        state: The target state of circuit initialized to. For example: "0", "1", "+"

    """

    def __init__(self, state: Union[str, int]):
        super().__init__()
        self.state = state

    def __mul__(self, qureg: Qureg):
        """Apply the QSP operator."""
        if not isinstance(qureg, Qureg):
            raise TypeError("the operand must be Qureg.")

        if not self._check_state(qureg):
            raise ValueError(f"Invalid ('{self.state}') state.")

        self._process_state(qureg)

    @abstractmethod
    def _check_state(self):
        pass

    @abstractmethod
    def _process_state(self):
        pass

