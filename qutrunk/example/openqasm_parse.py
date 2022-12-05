"""Deserialize OpenQASM file object."""

import os
from pathlib import Path

from qutrunk.circuit import QCircuit


def run_openqasm_parse():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    qasm_file = Path(cur_path) / "bell_pair.qasm"
    circuit = QCircuit.load(file=qasm_file, format="openqasm")

    # run circuit
    res = circuit.run(shots=100)

    # print result
    print(res.get_counts())

    return circuit


if __name__ == "__main__":
    circuit = run_openqasm_parse()
    circuit.draw()
