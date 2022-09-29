"""set amplitudes gate."""

import cmath
import numpy as np

from .basicgate import BasicGate
from qutrunk.circuit.command import Amplitude, Command, CmdEx
from qutrunk.circuit import Qureg

class AMP(BasicGate):
    """Apply set amplitudes gate.

    Example:
        .. code-block:: python

            AMP(reals, images, startind, numamps) * qr
    """

    def __init__(self, reals, images, startind, numamps):
        super().__init__()
        self.reals = reals
        self.images = images
        self.startind = startind
        self.numamps = numamps

    def __str__(self):
        return "AMP"

    def __mul__(self, qureg):
        """Quantum custom logic gate operation.

        Args:
            reals: amplitudes real part
            imags: amplitudes imag part
            startindex: amplitudes start index
            endindex: amplitudes end index

        """
        if not isinstance(qureg, Qureg):
            raise NotImplementedError("The argument must be Qureg object.")

        cmd = Command(self, cmdex=CmdEx(Amplitude()))
        cmd.cmdex.amp.reals = self.reals
        cmd.cmdex.amp.imags = self.images
        cmd.cmdex.amp.startind = self.startind
        cmd.cmdex.amp.numamps = self.numamps
        
        self.commit(qureg.circuit, cmd)

    @property
    def label(self):
        """A label for identifying the gate."""
        self.__str__()