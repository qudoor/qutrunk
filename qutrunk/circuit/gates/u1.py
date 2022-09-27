"""U1 Gate."""

import numpy as np

from .basicgate import BasicGate
from .basicgate import BasicRotateGate
from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit


class U1(BasicGate):
    """U1 gate.

    Args:
        alpha: Rotation angle.

    Example:
        .. code-block:: python

            U1(pi/2) * qr[0]
    """

    def __init__(self, alpha):
        """
        Args:
            alpha: Rotation angle.
        """
        super().__init__()
        self.rotation = alpha

    def __str__(self):
        return "U1"

    def __or__(self, qubit):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply U1 gate.

        Example:
            .. code-block:: python

                U1(pi/2) * qr[0]

        Raises:
            NotImplementedError: If the argument is not a Qubit object.
        """
        if not isinstance(qubit, QuBit):
            # TODO: need to improve.
            raise NotImplementedError("The argument must be Qubit object.")

        targets = [qubit.index]
        cmd = Command(self, targets, rotation=[self.rotation], inverse=self.is_inverse)
        self.commit(qubit.circuit, cmd)
        return cmd

    def __mul__(self, qubit):
        """Overwrite * operator to achieve quantum logic gate operation, \
            reuse __or__ operator implement."""
        return self.__or__(qubit)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        lam = float(self.rotation)
        return np.array([[1, 0], [0, np.exp(1j * lam)]])


class CU1(BasicRotateGate):
    """Control U1 gate.

    Args:
        alpha: Rotation angle.

    Example:
        .. code-block:: python

           CU1(pi/2) * (qr[0], qr[1])
    """

    def __init__(self, alpha):
        """Create new CU1 gate.
        Args:
            alpha: Rotation angle.
        """
        super().__init__()
        self.rotation = alpha

    def __str__(self):
        return "CU1"

    def __or__(self, qubits):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply CU1 gate.

        Example:
            .. code-block:: python

                CU1(pi/2) * (qr[0], qr[1])

        Raises:
            NotImplementedError: If the argument is not a Qubit object.
            AttributeError: If the qubits should not be two.
        """
        if not all(isinstance(qubit, QuBit) for qubit in qubits):
            # TODO: need to improve.
            raise NotImplementedError("The argument must be Qubit object.")

        if len(qubits) != 2:
            # TODO: need to improve.
            raise AttributeError("Parameter Error: qubits should be two.")

        controls = [qubits[0].index]
        targets = [qubits[1].index]
        cmd = Command(
            self, targets, controls, inverse=self.is_inverse, rotation=[self.rotation]
        )
        self.commit(qubits[0].circuit, cmd)
        return cmd

    def __mul__(self, qubits):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        return self.__or__(qubits)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        # TODO: definition have problem.
        half_alpha = float(self.rotation)
        return np.matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, np.exp(1j * half_alpha)]]
        )
