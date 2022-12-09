import numpy as np

from .basicgate import BasicGate
from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit


class X1Gate(BasicGate):
    """Apply the single-qubit X1 gate.

    Example:
        .. code-block:: python

            X1 * qr[0]
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "X1"

    def __or__(self, qubit):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply X1 gate.

        Example:
            .. code-block:: python

                X1 * qr[0]

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
        factor = 1 / np.sqrt(2)
        return np.array([[factor, -1j * factor], [-1j * factor, factor]])

    def inv(self):
        """Apply inverse gate."""
        gate = X1Gate()
        gate.is_inverse = not self.is_inverse
        return gate


X1 = X1Gate()
