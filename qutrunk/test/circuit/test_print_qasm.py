import pytest
import math
import sys
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, CNOT


# class test_qasm(QCircuit):
def test_print_qasm():
    # qc = QCircuit(backend=backend)
    # qc.print_qasm(file="a.qasm")
    # with open(r"D:\projects\qubox\src\qutrunk\example\a.qasm","r") as stream:
    #     container = stream.read()
    # while True:
    #     lines=stream.readline()
    #     print(line)
    #     if not line:
    #         break
    qc = QCircuit()
    qreg = qc.allocate(2)
    # creg = qc.allocate(2)
    # apply gate


    # apply gate

    H * qreg[0]
    CNOT * (qreg[0], qreg[1])
    Measure * qreg[0]
    Measure * qreg[1]
    # circuit_in = circuit(qr, cr)
    # qc.print_qasm()
    # for c in self:
    #     f.write(c.qasm() + ";\n")
    # circuit_in = ('include "qulib1.inc";\n', f"qreg q[{str(len(qreg))}];\n", f"creg c[{str(len(qreg))}];\n")

    # circuit_in.h(qreg[0])
    # circuit_in.h(qreg[1])
    # circuit_in.measure(qreg[0], creg[0])
    # circuit_in.measure(qreg[1], creg[1])
    # circuit_in.x(qreg[0]).c_if(creg, 0x3)
    # circuit_in.measure(qreg[0], creg[0])
    # circuit_in.measure(qreg[1], creg[1])
    # circuit_in.measure(qreg[2], creg[2])
    # aa = sys.stdout.readlines()

    qc.print_qasm(file="b.qasm")


    # circuit_in = ["""include "qulib1.inc";""","qreg q["+str(len(qreg))+"];","creg c["+str(len(qreg))+"];","h q[0]","cx q[0],q[1];","measure q[0] -> c[0];","measure q[1] -> c[1];"]

#     expected_qasm = """OPENQASM 2.0;
# include "qulib1.inc";
# qreg q[2];
# creg c[2];
# h q[0];
# cx q[0],q[1];
# measure q[0] -> c[0];
# measure q[1] -> c[1];
# """
    # print(expected_qasm)
    # circuit_in[1] = f"creg c[{str(len(self.qreg))}];\n"

    circuit_in = ['OPENQASM 2.0;\n', 'include "qulib1.inc";\n', 'qreg q[2];\n', 'creg c[2];\n', 'h q[0];\n', 'cx q[0],q[1];\n', 'measure q[0] -> c[0];\n', 'measure q[1] -> c[1];\n']

    with open(r"D:\projects\qutrunk\qutrunk\test\circuit\b.qasm", "r") as stream:
        container = stream.readlines()
        circuit_out = container
    # print(circuit_in)
    # assert circuit_out == circuit_in
    assert circuit_out == circuit_in


if __name__ == "__main__":
    test_print_qasm()
