"""Phase Gate."""

import numpy as np

from .basicgate import BasicPhaseGate
from .basicgate import BasicRotateGate
from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit


class P(BasicPhaseGate):
    """Phase gate.

    Args:
        alpha: The phase to apply.

    Example:
        .. code-block:: python

            P(alpha) * qr[0]
    """

    def __init__(self, alpha):
        if alpha is None:
            raise ValueError("The argument cannot be empty.")
        super().__init__()
        self.rotation = alpha

    def __str__(self):
        return "P"

    def __or__(self, qubit):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply phase gate.

        Example:
            .. code-block:: python

                P(alpha) * qr[0]

        Raises:
            TypeError: If the argument is not a Qubit object.
        """
        if not isinstance(qubit, QuBit):
            raise TypeError("The argument must be Qubit object.")

        targets = [qubit.index]
        cmd = Command(self, targets, rotation=[self.rotation], inverse=self.is_inverse)
        self.commit(qubit.circuit, cmd)

    def __mul__(self, qubit):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        self.__or__(qubit)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        return np.array([[1, 0], [0, np.exp(1j * self.rotation)]])

    def inv(self):
        """Apply inverse gate."""
        gate = P(self.rotation)
        gate.is_inverse = not self.is_inverse
        return gate

    def ctrl(self, ctrl_cnt=1):
        """Apply controlled gate.

        Args:
            ctrl_cnt: The number of control qubits, default: 1.
        """
        if ctrl_cnt > 1:
            raise ValueError("P gate do not support multiple control bits.")
        gate = CP(self.rotation)
        gate.is_inverse = self.is_inverse
        return gate


class CP(BasicRotateGate):
    """Control Phase Gate.

    Args:
        angle: The phase to apply.

    Example:
        .. code-block:: python

            CP * (qr[0], qr[1])
    """

    def __init__(self, angle):
        if angle is None:
            raise ValueError("The argument cannot be empty.")
        super().__init__()
        self.rotation = angle

    def __str__(self):
        return "CP"

    def __or__(self, qubits):
        """Quantum logic gate operation.

        Args:
            qubits: qubits[0] is control qubit, qubits[1] is target qubit.

        Example:
            .. code-block:: python

                CP * (qr[0], qr[1])

        Raises:
            NotImplementedError: If the argument is not a Qubit object.
        """
        if not all(isinstance(qubit, QuBit) for qubit in qubits):
            raise TypeError("The argument must be Qubit object.")

        if len(qubits) != 2:
            raise ValueError(
                "Parameter error：One controlled and one target qubit are required."
            )

        self.qubits = qubits
        controls = [qubits[0].index]
        targets = [qubits[1].index]
        cmd = Command(
            self, targets, controls, rotation=[self.rotation], inverse=self.is_inverse
        )
        self.commit(qubits[0].circuit, cmd)

    def __mul__(self, qubits):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        self.__or__(qubits)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        eith = np.exp(1j * float(self.rotation))
        # TODO: ??
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, eith]])

    def inv(self):
        """Apply inverse gate."""
        gate = CP(self.rotation)
        gate.is_inverse = not self.is_inverse
        return gate
