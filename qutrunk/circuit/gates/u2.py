"""U2 gate."""
import numpy as np

from .basicgate import BasicGate
from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit


class U2(BasicGate):
    """U2 gate.

    Args:
        theta: U2 gate parameter1.
        phi: U2 gate parameter2.

    Example:
        .. code-block:: python

            U2(pi/2, pi/2) * qr[0]
    """

    def __init__(self, theta, phi):
        """
        Args:
            theta: U2 gate parameter1.
            phi: U2 gate parameter2.
        """
        super().__init__()
        self.theta = theta
        self.phi = phi

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
            NotImplementedError: If the argument is not a Qubit object.
        """
        if not isinstance(qubit, QuBit):
            raise NotImplementedError("The argument must be Qubit object.")

        targets = [qubit.index]
        cmd = Command(
            self, targets, rotation=[self.theta, self.phi], inverse=self.is_inverse
        )
        self.commit(qubit.circuit, cmd)
        return cmd

    def __mul__(self, qubit):
        """Overwrite * operator to achieve quantum logic gate operation, \
            reuse __or__ operator implement."""
        return self.__or__(qubit)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        isqrt2 = 1 / np.sqrt(2)
        phi = self.theta
        lam = self.phi
        return np.matrix(
            [
                [isqrt2, -np.exp(1j * lam) * isqrt2],
                [np.exp(1j * phi) * isqrt2, np.exp(1j * (phi + lam)) * isqrt2],
            ]
        )
