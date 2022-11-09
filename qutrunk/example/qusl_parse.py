"""Deserialize QuSL file object."""

from pathlib import Path

from qutrunk.circuit import QCircuit


def run_qusl_parse():
    qusl_file = Path.cwd() / "bell_pair.qusl"
    circuit = QCircuit.load(file=qusl_file)

    # run circuit
    res = circuit.run(shots=100)

    # print result
    print(res.get_measure())
    print(res.get_counts())

    return circuit


if __name__ == "__main__":
    circuit = run_qusl_parse()
    circuit.draw()
