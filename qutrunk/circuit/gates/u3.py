"""U3 gate."""

import numpy as np

from .basicgate import BasicGate
from .basicgate import BasicRotateGate
from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit


class U3(BasicGate):
    """U3 gate.

    Args:
        theta: U3 gate parameter1.
        phi: U3 gate parameter2.
        lam: U3 gate parameter3.

    Example:
        .. code-block:: python

            U3(pi/2,pi/2,pi/2) * qr[0]
    """

    def __init__(self, theta, phi, lam):
        """Create new U3 gate.

        Args:
            theta: U3 gate parameter1.
            phi: U3 gate parameter2.
            lam: U3 gate parameter3.
        """
        super().__init__()

        if theta is None or phi is None or lam is None:
            raise ValueError("The argument cannot be empty.")
        self.theta = theta
        self.phi = phi
        self.lam = lam

    def __str__(self):
        return "U3"

    def __or__(self, qubit):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply U3 gate.

        Example:
            .. code-block:: python

                U3(pi/2,pi/2,pi/2) * qr[0]

        Raises:
            NotImplementedError: If the argument is not a Qubit object.
        """
        if not isinstance(qubit, QuBit):
            raise TypeError("The argument must be Qubit object.")

        targets = [qubit.index]
        cmd = Command(
            self,
            targets,
            rotation=[self.theta, self.phi, self.lam],
            inverse=self.is_inverse,
        )
        self.commit(qubit.circuit, cmd)

    def __mul__(self, qubit):
        """Overwrite * operator to achieve quantum logic gate operation, \
            reuse __or__ operator implement."""
        self.__or__(qubit)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        theta = self.theta
        phi = self.phi
        lam = self.lam
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        return np.array(
            [
                [cos, -np.exp(1j * lam) * sin],
                [np.exp(1j * phi) * sin, np.exp(1j * (phi + lam)) * cos],
            ]
        )

    def inv(self):
        """Apply inverse gate."""
        gate = U3(self.theta, self.phi, self.lam)
        gate.is_inverse = not self.is_inverse
        return gate

    def ctrl(self, ctrl_cnt=1):
        """Apply controlled gate.
        
        Args:
            ctrl_cnt: The number of control qubits, default: 1.
        """
        if ctrl_cnt > 1:
            raise ValueError("U3 gate do not support multiple control bits.")
        gate = CU3(self.theta, self.phi, self.lam)
        gate.is_inverse = self.is_inverse
        return gate


class CU3(BasicRotateGate):
    """Control U3 gate.

    Args:
            theta: U3 gate parameter 1.
            phi: U3 gate parameter 2.
            lam: U3 gate parameter 3.

    Example:
        .. code-block:: python

            CU3(pi/2,pi/2,pi/2) * (qr[0], qr[1])
    """

    def __init__(self, theta, phi, lam):
        """Create new CU3 gate.

        Args:
            theta: U3 gate parameter 1.
            phi: U3 gate parameter 2.
            lam: U3 gate parameter 3.
        """
        if theta is None or phi is None or lam is None:
            raise ValueError("The argument cannot be empty.")
        super().__init__()
        self.theta = theta
        self.phi = phi
        self.lam = lam

    def __str__(self):
        return "CU3"

    def __or__(self, qubits):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply CU3 gate.

        Example:
            .. code-block:: python

                CU3(pi/2,pi/2,pi/2) * (qr[0], qr[1])

        Raises:
            TypeError: If the argument is not a Qubit object.
            ValueError: If the qubits should not be two.
        """
        if not all(isinstance(qubit, QuBit) for qubit in qubits):
            raise TypeError("The argument must be Qubit object.")

        if len(qubits) != 2:
            raise ValueError("Parameter Error: One controlled and one target qubit are required.")

        controls = [qubits[0].index]
        targets = [qubits[1].index]
        cmd = Command(
            self,
            targets,
            controls,
            inverse=self.is_inverse,
            rotation=[self.theta, self.phi, self.lam],
        )
        self.commit(qubits[0].circuit, cmd)

    def __mul__(self, qubits):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        self.__or__(qubits)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        # TODO: definition have problem.
        half_alpha = float(self.theta)
        half_beta = float(self.phi)
        half_theta = float(self.lam)
        cos = np.cos(half_alpha / 2)
        sin = np.sin(half_alpha / 2)
        return np.array(
            [
                [1, 0, 0, 0],
                [0, cos, 0, -np.exp(1j * half_theta) * sin],
                [0, 0, 1, 0],
                [
                    0,
                    np.exp(1j * half_beta) * sin,
                    0,
                    np.exp(1j * (half_beta + half_theta)) * cos,
                ],
            ]
        )

    def inv(self):
        """Apply inverse gate."""
        gate = CU3(self.theta, self.phi,  self.lam)
        gate.is_inverse = not self.is_inverse
        return gate


class CU(BasicRotateGate):
    """Control U gate.

    Args:
        theta: U gate parameter 1.
        phi: U gate parameter 2.
        lam:U gate parameter 3.
        gamma: U gate parameter 4.

    Example:
        .. code-block:: python

            CU(pi/2,pi/2,pi/2,pi/2) * (qr[0], qr[1])
    """

    def __init__(self, theta, phi, lam, gamma):
        """
        Args:
            theta: U gate parameter 1.
            phi: U gate parameter 2.
            lam:U gate parameter 3.
            gamma: U gate parameter 4.
        """
        if theta is None or phi is None or lam is None or gamma is None:
            raise ValueError("The argument cannot be empty.")
        super().__init__()
        self.theta = theta
        self.phi = phi
        self.lam = lam
        self.gamma = gamma

    def __str__(self):
        return "CU"

    def __or__(self, qubits):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply CU gate.

        Example:
            .. code-block:: python

                CU(pi/2,pi/2,pi/2,pi/2) * (qr[0], qr[1])
        """
        if not all(isinstance(qubit, QuBit) for qubit in qubits):
            raise TypeError("The argument must be Qubit object.")

        if len(qubits) != 2:
            raise ValueError("Parameter error: One controlled and one target qubit is required.")

        self.qubits = qubits
        controls = [qubits[0].index]
        targets = [qubits[1].index]
        cmd = Command(
            self,
            targets,
            controls,
            inverse=self.is_inverse,
            rotation=[self.theta, self.phi, self.lam, self.gamma],
        )
        self.commit(qubits[0].circuit, cmd)

    def __mul__(self, qubits):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        self.__or__(qubits)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        half_theta = float(self.theta)
        half_phi = float(self.phi)
        half_lam = float(self.lam)
        half_gamma = float(self.gamma)
        cos = np.cos(half_theta / 2)
        sin = np.sin(half_theta / 2)
        a = np.exp(1j * half_gamma) * cos
        b = -np.exp(1j * (half_gamma + half_lam)) * sin
        c = np.exp(1j * (half_gamma + half_phi)) * sin
        d = np.exp(1j * (half_gamma + half_phi + half_lam)) * cos
        return np.array(
            [
                [1, 0, 0, 0],
                [0, a, 0, b],
                [0, 0, 1, 0],
                [0, c, 0, d],
            ]
        )

    def inv(self):
        """Apply inverse gate"""
        gate = CU(self.theta, self.phi, self.lam, self.gamma)
        gate.is_inverse = not self.is_inverse
        return gate
