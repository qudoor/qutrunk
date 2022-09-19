from .basicgate import BasicGate
from qutrunk.circuit.qubit import QuBit
from qutrunk.circuit.command import Command


class BarrierGate(BasicGate):
    """Barrier Gate.

    Args:
        BasicGate: Base class of all gates.

    Example:
        .. code-block:: python

            Barrier * (qr[0], qr[1])
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Barrier"

    def __or__(self, qubits):
        """Quantum logic gate operation.

        Args:
            qubits: The quantum bit to apply Barrier gate.

        Example:
            .. code-block:: python

                Barrier * (qr[0], qr[1])
        """
        qubit = [qubits] if isinstance(qubits, QuBit) else qubits
        targets = [q.index for q in qubit]
        cmd = Command(self, targets)
        self.commit(qubit[0].circuit, cmd)

    def __mul__(self, qubits):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        self.__or__(qubits)

    @property
    def label(self):
        """Return gate label."""
        return "Barrier"


Barrier = BarrierGate()
