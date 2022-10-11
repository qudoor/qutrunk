import cmath

import numpy as np

from .basicgate import BasicRotateGate
from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit


class Rzz(BasicRotateGate):
    """RotationZZ gate class.

    Args:
        alpha: The angle to rotate.

    Example:
        .. code-block:: python

            Rzz(alpha) * (qr[0], qr[1])
    """

    def __init__(self, alpha):
        if alpha is None:
            raise NotImplementedError("The argument cannot be empty.")
        super().__init__()
        self.rotation = alpha

    def __str__(self):
        return "Rzz"

    def __or__(self, qubits):
        """Quantum logic gate operation.

        Args:
            qubits: The quantum bits to apply Rzz gate.

        Example:
            .. code-block:: python

                Rzz(alpha) * (qr[0], qr[1])

        Raises:
            NotImplementedError: If the argument is not a Qubit object.
            AttributeError: If the qubits should not be two.
        """
        if not all(isinstance(qubit, QuBit) for qubit in qubits):
            raise NotImplementedError("The argument must be Qubit object.")

        if len(qubits) != 2:
            raise AttributeError()
        targets = [q.index for q in qubits]
        cmd = Command(self, targets, rotation=[self.rotation], inverse=self.is_inverse)
        self.commit(qubits[0].circuit, cmd)

    def __mul__(self, qubits):
        """Overwrite * operator to achieve quantum logic gate operation, \
            reuse __or__ operator implement."""
        self.__or__(qubits)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        return np.matrix(
            [
                [cmath.exp(-0.5 * 1j * self.rotation), 0, 0, 0],
                [0, cmath.exp(0.5 * 1j * self.rotation), 0, 0],
                [0, 0, cmath.exp(0.5 * 1j * self.rotation), 0],
                [0, 0, 0, cmath.exp(-0.5 * 1j * self.rotation)],
            ]
        )
