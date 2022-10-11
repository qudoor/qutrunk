"""Bernstein-Vazirani example."""

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import CNOT, X


def run_bernstein_vazirani(backend=None):
    num_qubits = 9
    secret_num = 2**4 + 1
    circuit = QCircuit(backend=backend, resource=True)

    qureg = circuit.allocate(num_qubits)

    X * qureg[0]

    bits = secret_num
    bit = 0
    for qb in range(1, num_qubits):
        bit = int(bits % 2)
        bits = int(bits / 2)
        if bit:
            CNOT * (qureg[0], qureg[qb])

    success_prob = 1.0
    bits = secret_num
    for qb in range(1, num_qubits):
        bit = int(bits % 2)
        bits = int(bits / 2)
        success_prob *= circuit.get_prob_outcome(qb, bit)

    print(f"solution reached with probability {success_prob}")
    circuit.print()

    circuit.run()
    circuit.show_resource()

    return circuit


if __name__ == "__main__":
    # local run
    circuit = run_bernstein_vazirani()
    circuit.draw()
