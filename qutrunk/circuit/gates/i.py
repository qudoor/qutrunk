import numpy as np

from qutrunk.circuit import Command
from .basicgate import BasicGate, Observable, PauliType


class IGate(BasicGate, Observable):
    """Apply the single-qubit Identity gate.

    Example:
        .. code-block:: python

            I * qr[0]
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Id"

    def __or__(self, qubit):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply I gate.

        Example:
            .. code-block:: python

                I * qr[0]
        """
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
        return np.matrix([[1, 0], [0, 1]])

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
        pauli["oper_type"] = PauliType.POT_PAULI_I.value
        pauli["target"] = target.index
        puali_list.append(pauli)
        return puali_list


PauliI = I = IGate()
