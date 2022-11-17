"""Deserialize QuSL file object."""

import os
from pathlib import Path

from qutrunk.circuit import QCircuit


def run_qusl_parse():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    qusl_file = Path(cur_path) / "bell_pair.qusl"
    circuit = QCircuit.load(file=qusl_file)

    # run circuit
    res = circuit.run(shots=100)

    # print result
    print(res.get_measures())
    print(res.get_counts())

    return circuit


if __name__ == "__main__":
    circuit = run_qusl_parse()
    circuit.draw()
