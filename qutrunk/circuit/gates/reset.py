import numpy as np

from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit
from .basicgate import BasicGate


class ResetGate(BasicGate):
    """Reset qubits gate.

    Reset qubit to zero state.

    Example:
        .. code-block:: python

            Reset * q[0]
            Reset * (q[0], q[1])
            Reset * q
    """

    def __init__(self):
        super().__init__()
        self.name = str(self)

    def __str__(self):
        return "Reset"

    def __or__(self, qubit):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply Reset gate.

        Example:
            .. code-block:: python

                Reset * q[0]
                Reset * (q[0], q[1])
                Reset * q
        """
        qubits = [qubit] if isinstance(qubit, QuBit) else qubit
        targets = [q.index for q in qubits]
        cmd = Command(self, targets)
        self.commit(qubits[0].circuit, cmd)

    def __mul__(self, qubit):
        """Overwrite * operator to achieve quantum logic gate operation, \
            reuse __or__ operator implement."""
        self.__or__(qubit)

Reset = ResetGate()
