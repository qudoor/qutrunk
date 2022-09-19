from qutrunk.circuit import Command
from .basicgate import BasicGate


class MeasureGate(BasicGate):
    """Measures a single qubit, collapsing it randomly to 0 or 1.

    Example:
        .. code-block:: python

            Measure * qr[0]
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Measure"

    def __or__(self, qubit):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to be measured.

        Example:
            .. code-block:: python

                Measure * qr[0]
        """
        targets = [qubit.index]
        cmd = Command(self, targets)
        self.commit(qubit.circuit, cmd)

    def __mul__(self, qubit):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        self.__or__(qubit)


Measure = MeasureGate()
