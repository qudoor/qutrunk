"""U2 gate."""

import numpy as np

from .basicgate import BasicGate
from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit


class U2(BasicGate):
    """U2 gate, single-qubit rotation about the X+Z axis.
    
    Example:
        .. code-block:: python

            U2(pi/2, pi/2) * qr[0]
    """

    def __init__(self, phi, lam):
        if lam is None or phi is None:
            raise ValueError("The argument cannot be empty.")
        super().__init__()
        self.lam = lam
        self.phi = phi
        self.lam = lam

    def __str__(self):
        return "U2"

    def __or__(self, qubit):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply U2 gate.

        Example:
            .. code-block:: python

                U2(pi/2, pi/2) * qr[0]

        Raises:
            TypeError: If the argument is not a Qubit object.
        """
        if not isinstance(qubit, QuBit):
            raise TypeError("The argument must be Qubit object.")

        targets = [qubit.index]
        cmd = Command(
            self, targets, rotation=[self.phi, self.lam], inverse=self.is_inverse
        )
        self.commit(qubit.circuit, cmd)

    def __mul__(self, qubit):
        """Overwrite * operator to achieve quantum logic gate operation, \
            reuse __or__ operator implement."""
        self.__or__(qubit)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        isqrt2 = 1 / np.sqrt(2)
        phi = self.phi
        lam = self.lam
        return np.array(
            [
                [isqrt2, -np.exp(1j * lam) * isqrt2],
                [np.exp(1j * phi) * isqrt2, np.exp(1j * (phi + lam)) * isqrt2],
            ]
        )

    def inv(self):
        """Apply inverse gate."""
        gate = U2(self.phi, self.lam)
        gate.is_inverse = not self.is_inverse
        return gate
