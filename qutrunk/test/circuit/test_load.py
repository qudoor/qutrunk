"""Test the load function in QCircuit."""
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, CNOT


def test_load_qusl():
    """Test the functionality to deserialize a file object containing an qusl document into a Python object."""
    qc = QCircuit()
    qreg = qc.allocate(2)

    H * qreg[0]
    CNOT * (qreg[0], qreg[1])
    Measure * qreg[0]
    Measure * qreg[1]

    expect_in = qc.cmds

    circuit_in = qc.load(file="a.qusl", format="qusl")

    assert expect_in == circuit_in.cmds


def test_load_openqasm():
    """Test the functionality to deserialize a file object containing an OpenQASM document into a Python object."""
    qc = QCircuit()
    qreg = qc.allocate(2)

    H * qreg[0]
    CNOT * (qreg[0], qreg[1])
    Measure * qreg[0]
    Measure * qreg[1]

    expect_in = qc.cmds

    circuit_in = qc.load(file="b.qasm", format="openqasm")

    assert expect_in == circuit_in.cmds



