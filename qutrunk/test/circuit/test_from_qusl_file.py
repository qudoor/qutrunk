from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, CNOT

def test_qusl_file():
    qc = QCircuit()
    qreg = qc.allocate(2)
    H * qreg[0]
    CNOT * (qreg[0], qreg[1])
    Measure * qreg[0]
    Measure * qreg[1]
    qc.print(file="a.qusl")


    # a.print(file="d.qusl")

    # with open(r"D:\projects\qutrunk\qutrunk\test\circuit\d.qusl", "r") as stream:
    #     container = stream.readline()
    #
    # circuit_in = container[container.rfind("code") - 1:-1]
    # expect_in = '"code": ["H * q[0]\\n", "MCX * (q[0], q[1])\\n", "Measure * q[0]\\n", "Measure * q[1]\\n"]'

if __name__ == "__main__":
    test_qusl_file()