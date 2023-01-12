"""Grover's search algorithm."""

import math
import random
import time

from numpy import pi

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Measure, All
from qutrunk.circuit.ops import PLUS, QAA


def run_grover(qubits=10, backend=None):
    # Quantum qubits
    num_qubits = qubits

    # Number of amplitudes
    num_elems = 2**num_qubits

    # Count of iteration
    num_reps = math.ceil(pi / 4 * math.sqrt(num_elems))
    print("num_qubits:", num_qubits, "num_elems:", num_elems, "num_reps:", num_reps)

    # Choose target state randomly
    random.seed(int(time.time()))
    sol_elem = random.randint(0, num_elems - 1)
    print(f"target state: |{str(sol_elem)}>")

    # Create quantum circuit with local python simulator
    circuit = QCircuit(backend=backend, resource=True)

    # Allocate quantum qubits
    qureg = circuit.allocate(num_qubits)

    # Set inital amplitudes to plus state
    PLUS * qureg

    # Apply quantum operator(gates)
    QAA(num_reps, sol_elem) * qureg

    # Measure for all qubits
    All(Measure) * qureg

    # Run circuit in local simulator
    res = circuit.run()

    # Get measure result and print as int
    outlist = res.get_bitstrs()
    for out in outlist:
        print("measure result: " + str(int(out, base=2)))

    # Print quantum circuit resource information
    # circuit.show_resource()

    # Print quantum circuit execution information
    # print(res.running_info())

    return circuit


if __name__ == "__main__":
    # Run locally
    circuit = run_grover()
