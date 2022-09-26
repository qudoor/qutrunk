"""Test the function of calculating the depth of quantum circuits."""
from numpy import pi

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, CNOT, iSwap, T, Swap, SqrtSwap, CH, All


def test_depth():
    """Test the depth function in QCircuit."""
    qc = QCircuit()
    qreg = qc.allocate(5)

    H * qreg[0]
    H * qreg[1]
    H * qreg[2]
    iSwap(pi / 2) * (qreg[0], qreg[4])
    T * qreg[1]
    H * qreg[0]
    Swap * (qreg[1], qreg[4])
    CH * (qreg[0], qreg[1])
    SqrtSwap * (qreg[2], qreg[4])
    T * qreg[0]
    H * qreg[1]
    CNOT * (qreg[0], qreg[2])
    CH * (qreg[1], qreg[2])
    H * qreg[2]
    All(Measure) * qreg

    result = qc.depth()
    calculate = 9

    assert result == calculate





