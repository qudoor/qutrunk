"""Quantum Phase Estimation."""

from qutrunk.circuit.ops.operator import Operator, OperatorContext
from math import pi
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, NOT, Measure, CP, All, Barrier
from qutrunk.circuit.ops import IQFT


class QPE(Operator):
    def __init__(self, unitary):
        """Initialize a QPE gate."""
        super().__init__()
        self.unitary = unitary

    def __str__(self):
        """Return a string representation of the object."""
        return f"QPE({str(self.unitary)})"

    def __mul__(self):
        raise NotImplementedError
