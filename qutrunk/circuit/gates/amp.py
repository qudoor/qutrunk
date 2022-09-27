"""set amplitudes gate."""

import cmath
import numpy as np

from .basicgate import BasicGate
from qutrunk.circuit import Command, Qureg

class AMP(BasicGate):
    """Apply set amplitudes gate.

    Example:
        .. code-block:: python

            AMP(reals, images, ampstartind, numamps) * qr
    """

    def __init__(self, reals, images, ampstartind, numamps):
        super().__init__()
        self.reals = reals
        self.images = images
        self.ampstartind = ampstartind
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

        cmd = Command(self)
        cmd.reals = self.reals
        cmd.imags = self.images
        cmd.ampstartind = self.ampstartind
        cmd.numamps = self.numamps
        
        self.commit(qureg.circuit, cmd)

    @property
    def label(self):
        """A label for identifying the gate."""
        self.__str__()