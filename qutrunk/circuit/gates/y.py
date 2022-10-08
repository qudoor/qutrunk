import numpy as np

from qutrunk.circuit import Command
from .basicgate import BasicGate, Observable, PauliType
from qutrunk.circuit.qubit import QuBit


class YGate(BasicGate, Observable):
    """Apply the single-qubit Pauli-Y (also known as the Y or sigma-Y) gate.

    Example:
        .. code-block:: python

            Y * qr[0]
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Y"

    def __or__(self, qubit):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply Y gate.

        Example:
            .. code-block:: python

                Y * qr[0]

        Raises:
            NotImplementedError: If the argument is not a Qubit object.
        """
        if not isinstance(qubit, QuBit):
            raise NotImplementedError("The argument must be Qubit object.")

        targets = [qubit.index]
        cmd = Command(self, targets, inverse=self.is_inverse)
        self.commit(qubit.circuit, cmd)

    def __mul__(self, qubit):
        """Overwrite * operator to achieve quantum logic gate operation, \
            reuse __or__ operator implement."""
        self.__or__(qubit)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        return np.array([[0, -1j], [1j, 0]])

    def obs(self, target):
        """Get Observable data.

        Args:
            target: The observed qubit.

        Returns:
            The observed data list, each item contains op type and target qubit, \
                e.g: [{"oper_type": 1, "target": 0}].
        """
        puali_list = []
        pauli = {}
        pauli["oper_type"] = PauliType.POT_PAULI_Y.value
        pauli["target"] = target.index
        puali_list.append(pauli)
        return puali_list

    def inv(self):
        """Apply inverse gate"""
        gate = YGate()
        gate.is_inverse = not self.is_inverse
        return gate

    def ctrl(self, ctrl_cnt=1):
        """Apply controlled gate.
        
        Args:
            ctrl_cnt: The number of control qubits, default: 1.
        """
        if ctrl_cnt > 1:
            raise ValueError("Y gate do not support multiple control bits.")
        gate = CYGate()
        gate.is_inverse = self.is_inverse
        return gate


PauliY = Y = YGate()


class CYGate(BasicGate):
    """Control Y gate.

    Example:
        .. code-block:: python

            CY * (qr[0], qr[1])
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "CY"

    def __or__(self, qubits):
        """Quantum logic gate operation.

        Args:
            qubits: qubits[0] is control qubit, qubits[1] is target qubit.

        Example:
            .. code-block:: python

                CY * (qr[0], qr[1])
        """
        if len(qubits) != 2:
            raise AttributeError(
                "Argument error：need to one controlled qubit and one target qubit."
            )
        self.qubits = qubits
        controls = [qubits[0].index]
        targets = [qubits[1].index]
        cmd = Command(self, targets, controls, inverse=self.is_inverse)
        self.commit(qubits[0].circuit, cmd)

    def __mul__(self, qubits):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        self.__or__(qubits)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        return np.array([[0, 0, -1j, 0], [0, 1, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 1]])

    def inv(self):
        """Apply inverse gate"""
        gate = CYGate()
        gate.is_inverse = not self.is_inverse
        return gate


CY = CYGate()
