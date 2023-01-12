"""U1 Gate."""

import numpy as np

from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit
from .basicgate import BasicGate, BasicRotateGate


class U1(BasicGate):
    """U1 gate, single-qubit rotation about the Z axis.

    Example:
        .. code-block:: python

            U1(pi/2) * qr[0]
    """

    def __init__(self, lam):
        """
        Args:
            alpha: Rotation angle.
        """
        if lam is None:
            raise NotImplementedError("The argument cannot be empty.")
        super().__init__()
        self.lam = lam

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
            TypeError: If the argument is not a Qubit object.
        """
        if not isinstance(qubit, QuBit):
            raise TypeError("The argument must be Qubit object.")

        targets = [qubit.index]
        cmd = Command(self, targets, rotation=[self.lam], inverse=self.is_inverse)
        self.commit(qubit.circuit, cmd)

    def __mul__(self, qubit):
        """Overwrite * operator to achieve quantum logic gate operation, \
            reuse __or__ operator implement."""
        self.__or__(qubit)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        lam = float(self.lam)
        return np.array([[1, 0], [0, np.exp(1j * lam)]])

    def inv(self):
        """Apply inverse gate."""
        gate = U1(self.lam)
        gate.is_inverse = not self.is_inverse
        return gate

    def ctrl(self, ctrl_cnt=1):
        """Apply controlled gate.
        
        Args:
            ctrl_cnt: The number of control qubits, default: 1.
        """
        if ctrl_cnt > 1:
            raise ValueError("U1 gate do not support multiple control bits.")
        gate = CU1(self.lam)
        gate.is_inverse = self.is_inverse
        return gate


class CU1(BasicRotateGate):
    """Control U1 gate.

    Example:
        .. code-block:: python

           CU1(pi/2) * (qr[0], qr[1])
    """

    def __init__(self, lam):
        """Create new CU1 gate.
        Args:
            lam: Rotation angle.
        """
        if lam is None:
            raise TypeError("The argument cannot be empty.")
        super().__init__()
        self.lam = lam

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
            TypeError: If the argument is not a Qubit object.
            AttributeError: If the qubits should not be two.
        """
        if not all(isinstance(qubit, QuBit) for qubit in qubits):
            raise TypeError("The argument must be Qubit object.")

        if len(qubits) != 2:
            raise ValueError("Parameter error: One controlled and one target qubit are required.")

        controls = [qubits[0].index]
        targets = [qubits[1].index]
        cmd = Command(
            self, targets, controls, inverse=self.is_inverse, rotation=[self.lam]
        )
        self.commit(qubits[0].circuit, cmd)

    def __mul__(self, qubits):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        self.__or__(qubits)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        # TODO: definition have problem.
        half_alpha = float(self.lam)
        return np.matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.exp(1j * half_alpha)]]
        )

    def inv(self):
        """Apply inverse gate."""
        gate = CU1(self.lam)
        gate.is_inverse = not self.is_inverse
        return gate
