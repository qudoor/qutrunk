"""The S gate."""

import numpy as np

from .basicgate import BasicGate
from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit


class SGate(BasicGate):
    """Apply the single-qubit S gate.

    Example:
        .. code-block:: python

            S * qr[0]
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "S"

    def __or__(self, qubit):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply S gate.

        Example:
            .. code-block:: python

                S * qr[0]

        Raises:
            NotImplementedError: If the argument is not a Qubit object.
        """
        if not isinstance(qubit, QuBit):
            raise TypeError("The argument must be Qubit object.")

        targets = [qubit.index]
        cmd = Command(self, targets, inverse=self.is_inverse)
        self.commit(qubit.circuit, cmd)

    def __mul__(self, qubit):
        """Overwrite * operator to achieve quantum logic gate operation, \
            reuse __or__ operator implement."""
        self.__or__(qubit)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        return np.array([[1, 0], [0, 1j]])

    def inv(self):
        """Apply inverse gate(SdgGate)."""
        gate = SGate()
        gate.is_inverse = not self.is_inverse
        return gate


S = SGate()
