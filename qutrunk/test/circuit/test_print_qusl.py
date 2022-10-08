"""Testing the functionality of exporting quantum circuits in Qusl format."""
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, CNOT


def test_print_qusl():
    """Test the print function in QCircuit."""
    qc = QCircuit()
    qreg = qc.allocate(2)

    H * qreg[0]
    CNOT * (qreg[0], qreg[1])
    Measure * qreg[0]
    Measure * qreg[1]

    qc.print(file="a.qusl")
    expect_out = '"code": ["H * q[0]\\n", "MCX * (q[0], q[1])\\n", "Measure * q[0]\\n", "Measure * q[1]\\n"]'

    with open("a.qusl", "r") as stream:
        container = stream.readline()
        circuit_out = container[container.rfind("code")-1:-1]

    assert circuit_out == expect_out
