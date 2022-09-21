"""Grover's search algorithm."""

import math
import random

from numpy import pi

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Measure, All
from qutrunk.circuit.ops import QSP, QAA


def run_grover(qubits=10, backend=None):
    num_qubits = qubits
    num_elems = 2**num_qubits
    num_reps = math.ceil(pi / 4 * math.sqrt(num_elems))
    print("num_qubits:", num_qubits, "num_elems:", num_elems, "num_reps:", num_reps)

    sol_elem = random.randint(0, num_elems - 1)
    print(f"target state: |{str(sol_elem)}>")

    circuit = QCircuit(backend=backend, resource=True)
    qureg = circuit.allocate(num_qubits)

    QSP("+") * qureg
    QAA(num_reps, sol_elem) * qureg

    All(Measure) * qureg

    res = circuit.run()
    out = res.get_outcome()
    print("measure result: " + str(int(out, base=2)))
    circuit.show_resource()
    print(res.excute_info())

    return circuit


if __name__ == "__main__":
    # local run
    circuit = run_grover()
