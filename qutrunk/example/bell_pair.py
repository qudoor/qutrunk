"""Bell state example."""
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, CNOT, Measure


def run_bell_pair(backend=None):
    # allocate
    qc = QCircuit(backend=backend)
    qr = qc.allocate(2)

    # apply gate
    H * qr[0]
    CNOT * (qr[0], qr[1])
    Measure * qr[0]
    Measure * qr[1]

    # print circuit
    qc.print()
    # qc.dump(file="bell_pair.qusl")

    # run circuit
    res = qc.run(shots=100)

    # print result
    print(res.get_measure())
    print(res.get_counts())
    # print(res.excute_info())

    return qc


if __name__ == "__main__":
    circuit = run_bell_pair()
    circuit.draw()

