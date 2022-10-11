"""Deserialize OpenQASM file object."""
from qutrunk.circuit import QCircuit


def run_openqasm_parse():
    circuit = QCircuit.load(file="bell_pair.qasm", format="openqasm")

    # run circuit
    res = circuit.run(shots=100)

    # print result
    print(res.get_measure())
    print(res.get_counts())

    return circuit


if __name__ == "__main__":
    circuit = run_openqasm_parse()
    circuit.draw()
