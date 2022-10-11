import numpy as np

from .basicgate import BasicRotateGate
from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit


class CR(BasicRotateGate):
    """Control rotation gate.

    Args:
        alpha: Rotation angle.

    Example:
        .. code-block:: python

            CR(pi/2) * (qr[0], qr[1])
    """

    def __init__(self, alpha):
        """
        Args:
            alpha: Rotation angle.
        """
        if alpha is None:
            raise NotImplementedError("The argument cannot be empty.")
        super().__init__()
        self.rotation = alpha

    def __str__(self):
        return "CR"

    def __or__(self, qubits):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply CR gate.

        Example:
            .. code-block:: python

                CR(pi/2) * (qr[0], qr[1])

        Raises:
            NotImplementedError: If the argument is not a Qubit object.
            AttributeError: If the qubits should not be two.
        """
        if not all(isinstance(qubit, QuBit) for qubit in qubits):
            raise NotImplementedError("The argument must be Qubit object.")

        if len(qubits) != 2:
            raise AttributeError("Parameter Error: qubits should be two.")
        self.qubits = qubits
        controls = [qubits[0].index]
        targets = [qubits[1].index]
        cmd = Command(
            self, targets, controls, inverse=self.is_inverse, rotation=[self.rotation]
        )
        self.commit(qubits[0].circuit, cmd)

    def __mul__(self, qubits):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        self.__or__(qubits)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        half_alpha = float(self.rotation)
        return np.matrix(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.exp(1j * half_alpha)],
            ]
        )
