import numpy as np

from .basicgate import BasicRotateGate
from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit


class Ry(BasicRotateGate):
    """Rotate a single qubit by a given angle around the Y-axis of the Bloch-sphere.

    Args:
        alpha: The angle to rotate.

    Example:
        .. code-block:: python

            Ry(alpha) * qr[0]
    """

    def __init__(self, alpha):
        if alpha is None:
            raise NotImplementedError("The argument cannot be empty.")
        super().__init__()
        self.rotation = alpha

    def __str__(self):
        return "Ry"

    def __or__(self, qubit):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply Ry gate.

        Example:
            .. code-block:: python

                Ry(alpha) * qr[0]

        Raises:
            NotImplementedError: If the argument is not a Qubit object.
        """
        if not isinstance(qubit, QuBit):
            # TODO:need to improve.
            raise NotImplementedError("The argument must be Qubit object.")

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
                [cos, -sin],
                [sin, cos],
            ]
        )


class CRy(BasicRotateGate):
    """Control Ry gate.

    Args:
        angle: The angle of the rotation.

    Example:
        .. code-block:: python

            CRy(pi/2) * (qr[0], qr[1])
    """

    def __init__(self, angle):
        if angle is None:
            raise NotImplementedError("The argument cannot be empty.")
        super().__init__()
        self.rotation = angle

    def __str__(self):
        return "CRy"

    def __or__(self, qubits):
        """Quantum logic gate operation.

        Args:
            qubits: qubits[0] is control qubit, qubits[1] is target qubit.

        Example:
            .. code-block:: python

            CRy(pi/2) * (qr[0], qr[1])
        """
        if not all(isinstance(qubit, QuBit) for qubit in qubits):
            # TODO:need to improve.
            raise NotImplementedError("The argument must be Qubit object.")

        if len(qubits) != 2:
            # TODO:need to improve.
            raise AttributeError(
                "Argument error：need to one controlled qubit and one target qubit."
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
        half_theta = float(self.rotation) / 2
        cos = np.cos(half_theta)
        sin = np.sin(half_theta)
        return np.array(
            [[cos, 0, -sin, 0], [0, 1, 0, 0], [sin, 0, cos, 0], [0, 0, 0, 1]]
        )
