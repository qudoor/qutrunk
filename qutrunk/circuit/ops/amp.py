import cmath
from typing import Optional

from qutrunk.circuit.command import Command, CmdEx, Amplitude
from qutrunk.circuit import Qureg
from qutrunk.circuit.ops import QSP


class AMP(QSP):
    """Quantum state preparation Operator.

    Init the quantum state to specific amplitude state.

    Args:
        classicvector: The amplitude state list.
        startind: The amplitude start index
        numamps: The number of amplitude

    Example:
        .. code-block:: python

            from qutrunk.circuit.ops import AMP
            from qutrunk.circuit import QCircuit
            from qutrunk.circuit.gates import H, All, Measure

            circuit = QCircuit()
            qureg = circuit.allocate(2)
            AMP([1-2j, 2+3j, 3-4j, 0.5+0.7j], 1, 2) * qureg
            print(circuit.get_statevector())
    """

    def __init__(
        self,
        classicvector: list,
        startind: Optional[int] = None,
        numamps: Optional[int] = None,
    ):
        super().__init__("AMP")
        self.classicvector = classicvector
        self.startind = startind
        self.numamps = numamps

    def __str__(self):
        return "AMP"

    def _check_state(self, qureg: Qureg):
        if self.startind is None:
            self.startind = 0

        if self.numamps is None or self.numamps > len(self.classicvector):
            self.numamps = len(self.classicvector)

        if (
            0 <= len(self.classicvector) <= 2 ** len(qureg)
            and 0 <= self.startind < 2 ** len(qureg)
            and 0 <= self.numamps <= 2 ** len(qureg)
            and (self.startind + self.numamps) <= 2 ** len(qureg)
        ):
            return True

        return False

    def _process_state(self, qureg: Qureg):
        """Process amp state."""
        reals = []
        imags = []
        listsum = sum(self.classicvector)
        for element in self.classicvector:
            normalized_element = cmath.sqrt(complex(element / listsum))
            reals.append(normalized_element.real)
            imags.append(normalized_element.imag)

        cmd = Command(self, cmdex=CmdEx(amp=Amplitude()))
        cmd.cmdex.amp.reals = reals
        cmd.cmdex.amp.imags = imags
        cmd.cmdex.amp.startind = self.startind
        cmd.cmdex.amp.numamps = self.numamps

        self.commit(qureg.circuit, cmd)

    def _append_statement(self, qureg: Qureg):
        qureg.circuit.append_statement(
            f"AMP({self.classicvector}, {self.startind}, {self.numamps}) * q"
        )
