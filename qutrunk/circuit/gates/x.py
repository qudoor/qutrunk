import numpy as np

from qutrunk.circuit import Command
from .basicgate import BasicGate, Observable, PauliType
from qutrunk.circuit.qubit import QuBit


class XGate(BasicGate, Observable):
    """Apply the single-qubit Pauli-X (also known as the X, sigma-X, NOT or bit-flip) gate.

    Example:
        .. code-block:: python

            X * qr[0]
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "X"

    def __or__(self, qubit):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply X gate.

        Example:
            .. code-block:: python

                X * qr[0]

        Raises:
            NotImplementedError: If the argument is not a Qubit object.
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
        return np.matrix([[0, 1], [1, 0]])

    def __call__(self, target):
        """
        Get Observable data.

        Args:
            target: The observed qubit.

        Returns:
            The observed data list, each item contains op type and target qubit, \
                e.g: [{"oper_type": 1, "target": 0}].
        """
        pauli = {}
        pauli["oper_type"] = PauliType.PAULI_X.value
        pauli["target"] = target.index
        return pauli

    def inv(self):
        """Apply inverse gate."""
        gate = XGate()
        gate.is_inverse = not self.is_inverse
        return gate

    def ctrl(self, ctrl_cnt=1):
        """Apply controlled gate.
        
        Args:
            ctrl_cnt: The number of control qubits, default: 1.
        """
        gate = MCX(ctrl_cnt)
        gate.is_inverse = self.is_inverse
        return gate


PauliX = X = NOT = XGate()


class MCX(BasicGate):
    """Multi-control X(NOT) gate.

    Args:
        ctrl_cnt: The number of control qubits.

    Example:
        .. code-block:: python

            MCX(2) * (qr[0], qr[1], qr[2]) # qr[0], qr[1] are control qubits, qr[2] is target qubit
            MCX(3) * (qr[0], qr[1], qr[2], qr[3])
    """

    def __init__(self, ctrl_cnt=1):
        """Create new MCX gate."""
        super().__init__()
        self.ctrl_cnt = ctrl_cnt

    def __str__(self):
        return "MCX"

    def __or__(self, qubits):
        """Quantum logic gate operation.

        Args:
            qubits: The left self.ctrl_cnt qubits are control qubits, the rest right bits are target qubits.

        Example:
            .. code-block:: python

                MCX(2) * (qr[0], qr[1], qr[2]) # qr[0], qr[1] are control qubits, qr[2] is target qubit
                MCX(3) * (qr[0], qr[1], qr[2], qr[3])
        """
        if not all(isinstance(qubit, QuBit) for qubit in qubits):
            raise TypeError("The argument must be Qubit object.")

        if len(qubits) <= self.ctrl_cnt:
            raise ValueError("The parameter miss controlled or target qubit(s).")

        self.qubits = qubits
        controls = [q.index for q in qubits[0 : self.ctrl_cnt]]
        targets = [q.index for q in qubits[self.ctrl_cnt :]]
        cmd = Command(self, targets, controls, inverse=self.is_inverse)
        self.commit(qubits[0].circuit, cmd)

    def __mul__(self, qubits):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        self.__or__(qubits)

    @property
    def matrix(self):
        """Access to the matrix property of this gate."""
        if self.ctrl_cnt == 1:
            return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    def inv(self):
        """Apply inverse gate."""
        gate = MCX(self.ctrl_cnt)
        gate.is_inverse = not self.is_inverse
        return gate


CX = CNOT = MCX(1)
Toffoli = MCX(2)
