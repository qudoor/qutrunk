import sys
import json
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, CNOT
from qutrunk.visualizations import circuit_drawer

from qutrunk.converters import dag_to_circuit



def test_qasm_file():

    qc = QCircuit()
    qreg = qc.allocate(2)
    H * qreg[0]
    CNOT * (qreg[0], qreg[1])
    Measure * qreg[0]
    Measure * qreg[1]
#     expect_in = """      ┌───┐      ┌─┐
# q[0]: ┤ H ├──■───┤M├───
#       └───┘┌─┴──┐└╥┘┌─┐
# q[1]: ─────┤ CX ├─╫─┤M├
#            └────┘ ║ └╥┘
#  c: 2/════════════╩══╩═
#                   0  1
#     """

    qc.print_qasm(file="b.qasm")
    a = qc.from_qasm_file(file=r'D:\projects\qutrunk\qutrunk\test\circuit\b.qasm')
    a.print(file='c.qasm')
    with open(r"D:\projects\qutrunk\qutrunk\test\circuit\c.qasm", "r") as stream:
        container = stream.readline()

    circuit_in = container[container.rfind("code")-1:-1]
    expect_in = '"code": ["H * q[0]\\n", "MCX * (q[0], q[1])\\n", "Measure * q[0]\\n", "Measure * q[1]\\n"]'
    # print(expect_in)
    # print(circuit_in)
    assert expect_in == circuit_in


    # a = str(qc.from_qasm_file(file=r'D:\projects\qutrunk\qutrunk\test\circuit\b.qasm'))
    # print(a)

    # b = qc.draw()
    # expect_in = str(qc.draw())
    # print(expect_in)
    # print(qc.from_qasm_file(file=r'D:\projects\qutrunk\qutrunk\test\circuit\b.qasm'))
    # p = str(qc.from_qasm_file)
    # print(qc.from_qasm_file)
    # print(expect_in)
    # print(qc.from_qasm_file)
    # dag_to_circuit()
    # assert expect_in == qc.from_qasm_file
    # print(p)
    # assert expect_in == a



if __name__ == "__main__":
    test_qasm_file()