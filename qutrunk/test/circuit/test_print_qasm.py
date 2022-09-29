"""Testing the functionality of exporting quantum circuits in OpenQASM format."""
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, CNOT


def test_print_qasm():
    """Test the print_qasm function in QCircuit."""
    qc = QCircuit()
    qreg = qc.allocate(2)

    H * qreg[0]
    CNOT * (qreg[0], qreg[1])
    Measure * qreg[0]
    Measure * qreg[1]

    qc.print_qasm(file="b.qasm")
    expect_out = ['OPENQASM 2.0;\n', 'include "qulib1.inc";\n', 'qreg q[2];\n', 'creg c[2];\n', 'h q[0];\n', 'cx q[0],q[1];\n', 'measure q[0] -> c[0];\n', 'measure q[1] -> c[1];\n']

    with open("b.qasm", "r") as stream:
        circuit_out = stream.readlines()

    assert circuit_out == expect_out
