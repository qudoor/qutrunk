"""qusl to circuit example."""
from qutrunk.circuit import QCircuit


def run_bell_pair(backend=None):
    # allocate
    qc = QCircuit.load(file="bell_pair.qusl", format="qusl")

    # print circuit
    qc.print()

    # run circuit
    res = qc.run(shots=100)

    # print result
    print(res.get_measure())

    return qc


if __name__ == "__main__":
    circuit = run_bell_pair()
    circuit.draw()

