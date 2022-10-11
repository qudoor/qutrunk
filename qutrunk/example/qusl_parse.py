"""Deserialize QuSL file object."""
from qutrunk.circuit import QCircuit


def run_qusl_parse():
    circuit = QCircuit.load(file="bell_pair.qusl")

    # run circuit
    res = circuit.run(shots=100)

    # print result
    print(res.get_measure())
    print(res.get_counts())

    return circuit


if __name__ == "__main__":
    circuit = run_qusl_parse()
    circuit.draw()
