from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Measure, CNOT


def test_print_qusl():

    qc = QCircuit()
    qreg = qc.allocate(2)
    H * qreg[0]
    CNOT * (qreg[0], qreg[1])
    Measure * qreg[0]
    Measure * qreg[1]

    qc.print(file="a.qusl")
    circuit_in = '"code": ["H * q[0]\\n", "MCX * (q[0], q[1])\\n", "Measure * q[0]\\n", "Measure * q[1]\\n"]'

    with open(r"D:\projects\qutrunk\qutrunk\test\circuit\a.qusl", "r") as stream:
        container = stream.readline()
        # circuit_out = container
    # print(circuit_out.rfind("code"))

    # print(circuit_out[circuit_out.rfind("code")-1:-1])
    circuit_out = container[container.rfind("code")-1:-1]
    print(circuit_in)
    print(circuit_out)
    assert circuit_out == circuit_in
    # f = open(r"D:\projects\qutrunk\qutrunk\test\circuit\a.qusl", "r")

    # 读取文件所有内容

    # print(f.seek(0,2))
    # # 关闭文件
    # f.close()



if __name__ == "__main__":
    test_print_qusl()
