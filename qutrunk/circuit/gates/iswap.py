import numpy as np

from .basicgate import BasicGate
from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit


class iSwapGate(BasicGate):
    """Performs a iSwap gate between qubit1 and qubit2.

    Example:
        .. code-block:: python

            iSwap * (qr[0], qr[1])
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "iSwap"

    def __or__(self, qubits):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to aplly iSwap gate.

        Example:
            .. code-block:: python

                iSwap * (qr[0], qr[1])

        Raises:
            NotImplementedError: If the argument is not a Qubit object.
            AttributeError: If the qubits should not be two.
        """
        if not all(isinstance(qubit, QuBit) for qubit in qubits):
            raise TypeError("The argument must be Qubit object.")

        if len(qubits) != 2:
            raise ValueError("Parameter Error: Two target bits are required.")

        self.qubits = qubits
        targets = [q.index for q in qubits]
        cmd = Command(self, targets, inverse=self.is_inverse)
        self.commit(qubits[0].circuit, cmd)

    def __mul__(self, qubits):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        self.__or__(qubits)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        # TODO: have problem.
        return np.matrix([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])

    def inv(self):
        """Apply inverse gate."""
        gate = iSwapGate()
        gate.is_inverse = not self.is_inverse
        return gate


iSwap = iSwapGate()
