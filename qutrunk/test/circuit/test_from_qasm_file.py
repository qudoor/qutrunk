from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, CNOT


def test_qasm_file():
    qc = QCircuit()
    qreg = qc.allocate(2)

    H * qreg[0]
    CNOT * (qreg[0], qreg[1])
    Measure * qreg[0]
    Measure * qreg[1]

    expect_in = qc.cmds

    result = qc.from_qasm_file(file='b.qasm')
    circuit_in = result.cmds

    assert expect_in == circuit_in

