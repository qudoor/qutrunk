import cmath

import numpy as np

from qutrunk.circuit import Command
from .basicgate import BasicGate


class Z1Gate(BasicGate):
    """Apply the single-qubit Z1 gate.

    Example:
        .. code-block:: python

            Z1 * qr[0]
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Z1"

    def __or__(self, qubit):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply Z1 gate.

        Example:
            .. code-block:: python

                Z1 * qr[0]
        """
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
        return np.matrix(
            [[np.exp(-1j * cmath.pi / 4), 0], [0, np.exp(1j * cmath.pi / 4)]]
        )

    def inv(self):
        """Apply inverse gate"""
        gate = Z1Gate()
        gate.is_inverse = bool(1-self.is_inverse)
        return gate


Z1 = Z1Gate()
