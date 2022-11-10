from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, CNOT


def test_qusl_file():
    qc = QCircuit()
    qreg = qc.allocate(2)

    H * qreg[0]
    CNOT * (qreg[0], qreg[1])
    Measure * qreg[0]
    Measure * qreg[1]

    expect_in = qc.cmds
    circuit_in = qc.load("bell_pair.qusl", "qusl")

    assert expect_in == circuit_in.cmds


