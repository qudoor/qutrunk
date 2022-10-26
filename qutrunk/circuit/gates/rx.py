import numpy as np

from .basicgate import BasicRotateGate
from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit


class Rx(BasicRotateGate):
    """Rotate a single qubit by a given angle around the X-axis of the Bloch-sphere.

    Args:
        alpha: The angle to rotate.

    Example:
        .. code-block:: python

            Rx(alpha) * qr[0]
    """

    def __init__(self, alpha):
        if alpha is None:
            raise ValueError("The argument cannot be empty.")

        super().__init__()
        self.rotation = alpha

    def __str__(self):
        return "Rx"

    def __or__(self, qubit):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to aplly Rx gate。

        Example:
            .. code-block:: python

                Rx(alpha) * qr[0]

        Raises:
            NotImplementedError: If the argument is not a Qubit object.
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
        half_theta = float(self.rotation) / 2
        cos = np.cos(half_theta)
        sin = np.sin(half_theta)
        return np.array(
            [
                [cos, -1j * sin],
                [-1j * sin, cos],
            ]
        )

    def inv(self):
        """Apply inverse gate"""
        gate = Rx(self.rotation)
        gate.is_inverse = not self.is_inverse 
        return gate

    def ctrl(self, ctrl_cnt=1):
        """Apply controlled gate.
        
        Args:
            ctrl_cnt: The number of control qubits, default: 1.
        """
        if ctrl_cnt > 1:
            raise ValueError("Rx gate do not support multiple control bits.")
        gate = CRx(self.rotation)
        gate.is_inverse = self.is_inverse
        return gate


class CRx(BasicRotateGate):
    """Control Rx Gate.

    Args:
        angle: The angle of the rotation.

    Example:
        .. code-block:: python

            CRx(pi/2) * (qr[0], qr[1])

    """

    def __init__(self, angle):
        if angle is None:
            raise ValueError("The argument cannot be empty.")

        super().__init__()
        self.rotation = angle

    def __str__(self):
        return "CRx"

    def __or__(self, qubits):
        """Quantum logic gate operation.

        Args:
            qubits: qubits[0] is control qubit, qubits[1] is target qubit.

        Example:
            .. code-block:: python

            CRx(pi/2) * (qr[0], qr[1])

        Raises:
            NotImplementedError: If the argument is not a Qubit object.
            AttributeError: If the qubits should not be two.
        """
        if not all(isinstance(qubit, QuBit) for qubit in qubits):
            raise TypeError("The argument must be Qubit object.")

        if len(qubits) != 2:
            raise ValueError(
                "Parameter error: One controlled and one target qubit are required."
            )

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
        # TODO:define have problem.
        half_theta = float(self.rotation) / 2
        cos = np.cos(half_theta)
        isin = 1j * np.sin(half_theta)
        return np.array(
            [[cos, 0, -isin, 0], [0, 1, 0, 0], [-isin, 0, cos, 0], [0, 0, 0, 1]]
        )

    def inv(self):
        """Apply inverse gate"""
        gate = CRx(self.rotation)
        gate.is_inverse = not self.is_inverse 
        return gate