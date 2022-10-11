"""Deserialize OpenQASM file object."""
import os
from pathlib import Path

from qutrunk.circuit import QCircuit


def run_openqasm_parse():
    qasm_file = Path(os.getcwd()) / "bell_pair.qasm"
    circuit = QCircuit.load(file=qasm_file, format="openqasm")

    # run circuit
    res = circuit.run(shots=100)

    # print result
    print(res.get_measure())
    print(res.get_counts())

    return circuit


if __name__ == "__main__":
    circuit = run_openqasm_parse()
    circuit.draw()
