import numpy as np

from .basicgate import BasicGate
from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit


class iSwap(BasicGate):
    """Performs a iSWAP gate between qubit1 and qubit2.

    Args:
        alpha: Rotation angle.

    Example:
        .. code-block:: python

            iSwap(pi/2) * (qr[0], qr[1])
    """

    def __init__(self, alpha):
        """
        Args:
            alpha: Rotation angle
        """
        if alpha is None:
            raise NotImplementedError("The argument cannot be empty.")
        super().__init__()
        self.rotation = alpha

    def __str__(self):
        return "iSwap"

    def __or__(self, qubits):
        """Quantum logic gate operation

        Args:
            qubit: The quantum bit to aplly iSwap gate.

        Example:
            .. code-block:: python

                iSwap(pi/2) * (qr[0], qr[1])

        Raises:
            NotImplementedError: If the argument is not a Qubit object.
            AttributeError: If the qubits should not be two.
        """
        if not all(isinstance(qubit, QuBit) for qubit in qubits):
            raise NotImplementedError("The argument must be Qubit object.")

        if len(qubits) != 2:
            raise AttributeError("Parameter Error: Two target bits are required.")

        self.qubits = qubits
        targets = [q.index for q in qubits]
        cmd = Command(self, targets, inverse=self.is_inverse, rotation=[self.rotation])
        self.commit(qubits[0].circuit, cmd)

    def __mul__(self, qubits):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        self.__or__(qubits)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        half_alpha = float(self.rotation)
        cos = np.cos(half_alpha)
        sin = np.sin(half_alpha)
        return np.matrix(
            [[1, 0, 0, 0], [0, cos, -1j * sin, 0], [0, -1j * sin, cos, 0], [0, 0, 0, 1]]
        )

    def inv(self):
        """Apply inverse gate"""
        gate = iSwap(self.rotation)
        gate.is_inverse = not self.is_inverse 
        return gate