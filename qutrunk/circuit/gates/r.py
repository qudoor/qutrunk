import math

import numpy as np

from .basicgate import BasicRotateGate
from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit


class R(BasicRotateGate):
    """Rotation alpha around the cos(beta)x + sin(beta)y axis.

    Args:
        theta: The angle to rotate.
        phi: Define the axis cos(beta)x + sin(beta)y.

    Example:
        .. code-block:: python

            R(theta, phi) * qr[0]
    """

    def __init__(self, theta, phi):
        if theta is None or phi is None:
            raise ValueError("The argument cannot be empty.")
        super().__init__()
        self.theta = theta
        self.phi = phi

    def __str__(self):
        return "R"

    def __or__(self, qubit):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply R gate.

        Example:
            .. code-block:: python

                R(theta, phi) * qr[0]

        Raises:
            TypeError: If the argument is not a Qubit object.
        """
        if not isinstance(qubit, QuBit):
            raise TypeError("The argument must be Qubit object.")

        targets = [qubit.index]
        cmd = Command(
            self, targets, rotation=[self.theta, self.phi], inverse=self.is_inverse
        )
        self.commit(qubit.circuit, cmd)

    def __mul__(self, qubit):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        self.__or__(qubit)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        # TODO:define have problem.
        theta, phi = float(self.theta), float(self.phi)
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        exp_m = np.exp(-1j * phi)
        exp_p = np.exp(1j * phi)
        return np.array([[cos, -1j * exp_m * sin], [-1j * exp_p * sin, cos]])

    def inv(self):
        """Apply inverse gate"""
        gate = R(self.theta, self.phi)
        gate.is_inverse = not self.is_inverse 
        return gate
