import numpy as np

from .basicgate import BasicGate
from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit


class Y1Gate(BasicGate):
    """Apply the single-qubit Y1 gate.

    Example:
        .. code-block:: python

            Y1 * qr[0]
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Y1"

    def __or__(self, qubit):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply Y1 gate.

        Example:
            .. code-block:: python

                Y1 * qr[0]

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
        return np.array([[factor, -1 * factor], [factor, factor]])

    def inv(self):
        """Apply inverse gate."""
        gate = Y1Gate()
        gate.is_inverse = not self.is_inverse
        return gate


Y1 = Y1Gate()
