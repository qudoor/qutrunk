"""Hadamard gate."""

import cmath

import numpy as np

from .basicgate import BasicGate
from qutrunk.circuit import Command
from qutrunk.circuit.qubit import QuBit


class HGate(BasicGate):
    """Apply the single-qubit Hadamard gate.

    Example:
        .. code-block:: python

            H * qr[0]
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "H"

    def __or__(self, qubit):
        """Quantum logic gate operation.

        Args:
            qubit: Quantum bit applied to hadamard gate.

        Example:
            .. code-block:: python

                H * qr[0]
        """
        if not isinstance(qubit, QuBit):
            raise TypeError("The argument must be Qubit object.")

        targets = [qubit.index]
        cmd = Command(self, targets, inverse=self.is_inverse)
        self.commit(qubit.circuit, cmd)

    def __mul__(self, qubit):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        self.__or__(qubit)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        return 1.0 / cmath.sqrt(2.0) * np.array([[1, 1], [1, -1]])

    @property
    def label(self):
        """A label for identifying the gate."""
        self.__str__()

    def inv(self):
        """Apply inverse gate."""
        gate = HGate()
        gate.is_inverse = not self.is_inverse 
        return gate

    def ctrl(self, ctrl_cnt=1):
        """Apply controlled gate.
        
        Args:
            ctrl_cnt: The number of control qubits, default: 1.
        """
        if ctrl_cnt > 1:
            raise ValueError("H gate do not support multiple control bits.")
        gate = CHGate()
        gate.is_inverse = self.is_inverse
        return gate


H = HGate()


class CHGate(BasicGate):
    """Controlled-Hadamard gate.

    Example:
        .. code-block:: python

            CH * (qr[0], qr[1])
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "CH"

    def __or__(self, qubits):
        """Quantum logic gate operation.

        Args:
            qubits: qubits[0] is control qubit, qubits[1] is target qubit.

        Example:
            .. code-block:: python

                CH * (qr[0], qr[1])

        Raises:
            AttributeError: If the argument is not a Qubit object.
        """
        if not all(isinstance(qubit, QuBit) for qubit in qubits):
            # TODO:need to improve.
            raise NotImplementedError("The argument must be Qubit object.")

        if len(qubits) != 2:
            # TODO:need to improve.
            raise AttributeError("Parameter error: One controlled and one target qubit are required.")

        self.qubits = qubits
        controls = [qubits[0].index]
        targets = [qubits[1].index]
        cmd = Command(self, targets, controls, inverse=self.is_inverse)
        self.commit(qubits[0].circuit, cmd)

    def __mul__(self, qubits):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement.

        Args:
            qubits: qubits[0] is control qubit, qubits[1] is target qubit.
        """
        self.__or__(qubits)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        _sqrt2o2 = 1 / cmath.sqrt(2)
        return np.array(
            [
                [_sqrt2o2, 0, _sqrt2o2, 0],
                [0, 1, 0, 0],
                [_sqrt2o2, 0, -_sqrt2o2, 0],
                [0, 0, 0, 1],
            ]
        )

    def inv(self):
        """Apply inverse gate."""
        gate = CHGate()
        gate.is_inverse = not self.is_inverse 
        return gate


CH = CHGate()
