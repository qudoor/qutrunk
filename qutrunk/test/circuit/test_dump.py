"""Test the dump function in QCircuit."""
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, CNOT


def test_dump_qusl():
    """Testing the functionality of serializing quantum circuits as QUSL files in JSON format"""
    qc = QCircuit()
    qreg = qc.allocate(2)

    H * qreg[0]
    CNOT * (qreg[0], qreg[1])
    Measure * qreg[0]
    Measure * qreg[1]

    qc.dump(file="a.qusl", format="qusl")
    expect_out = '"code": ["H * q[0]\\n", "MCX * (q[0], q[1])\\n", "Measure * q[0]\\n", "Measure * q[1]\\n"]'

    with open("a.qusl", "r") as stream:
        container = stream.readline()
        circuit_out = container[container.rfind("code") - 1:-1]

    assert circuit_out == expect_out


def test_dump_openqasm():
    """Testing the functionality of serializing quantum circuits as OpenQASM files in JSON format"""
    qc = QCircuit()
    qreg = qc.allocate(2)

    H * qreg[0]
    CNOT * (qreg[0], qreg[1])
    Measure * qreg[0]
    Measure * qreg[1]

    qc.dump(file="b.qasm", format="openqasm")
    expect_out = ['OPENQASM 2.0;\n', 'include "qulib1.inc";\n', 'qreg q[2];\n', 'creg c[2];\n', 'h q[0];\n',
                  'cx q[0],q[1];\n', 'measure q[0] -> c[0];\n', 'measure q[1] -> c[1];\n']

    with open("b.qasm", "r") as stream:
        circuit_out = stream.readlines()

    assert circuit_out == expect_out
