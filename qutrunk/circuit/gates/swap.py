"""Swap gate."""

import numpy as np

from .basicgate import BasicGate
from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit


class SwapGate(BasicGate):
    """Performs a SWAP gate between qubit1 and qubit2.

    Example:
        .. code-block:: python

            Swap * (qr[0], qr[1])
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Swap"

    def __or__(self, qubits):
        """Quantum logic gate operation.

        Args:
            qubits: Qubits to swap.

        Example:
            .. code-block:: python

                Swap * (qr[0], qr[1])

        Raises:
            NotImplementedError: If the argument is not a Qubit object.
        """
        if not all(isinstance(qubit, QuBit) for qubit in qubits):
            raise NotImplementedError("The argument must be Qubit object.")

        targets = [q.index for q in qubits]
        cmd = Command(self, targets, inverse=self.is_inverse)
        self.commit(qubits[0].circuit, cmd)
        return cmd

    def __mul__(self, qubits):
        """Overwrite * operator to achieve quantum logic gate operation, \
            reuse __or__ operator implement."""
        return self.__or__(qubits)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


Swap = SwapGate()


class CSwapGate(BasicGate):
    """Controlled-SWAP gate, also known as the Fredkin gate.

    Example:
        .. code-block:: python

            CSwap * (qr[0], qr[1], qr[2])
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "CSwap"

    def __or__(self, qubits):
        """Quantum logic gate operation.

        Args:
            qubits: qubits[0] is control qubit, qubits[1], qubits[2] is target qubits.

        Example:
            .. code-block:: python

                CSwap * (qr[0], qr[1], qr[2])
        """
        if not all(isinstance(qubit, QuBit) for qubit in qubits):
            raise NotImplementedError("The argument must be Qubit object.")

        if len(qubits) != 3:
            # TODO:need to improve.
            raise AttributeError(
                "Argument error：need to one controlled qubit and two target qubit."
            )

        self.qubits = qubits
        controls = [qubits[0].index]
        targets = [qubits[1].index, qubits[2].index]
        cmd = Command(self, targets, controls, inverse=self.is_inverse)
        self.commit(qubits[0].circuit, cmd)
        return cmd

    def __mul__(self, qubits):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        return self.__or__(qubits)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )


CSwap = CSwapGate()
