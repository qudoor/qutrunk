"""Bernstein-Vazirani example."""
import random

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import CNOT, X, H
from qutrunk.circuit.ops import Classical

PRECISION = 1e-10


def _apply_oracle(qr, num_qubits, secret):
    bits = secret
    for q in range(1, num_qubits):
        # extract the (q-1)-th bit of secret
        bit = int(bits % 2)
        bits = int(bits / 2)

        # NOT the ancilla, controlling on the q-th qubit
        if bit:
            CNOT * (qr[q], qr[0])


def apply_bernstein_vazirani(num_qubits, secret):
    # create circuit and qureg
    circuit = QCircuit()
    qr = circuit.allocate(num_qubits)

    # start in |0>
    Classical(0) * qr

    # NOT the ancilla
    X * qr[0]

    # H all qubits, including the ancilla
    for q in range(num_qubits):
        H * qr[q]

    _apply_oracle(qr, num_qubits, secret)

    # H all qubits, including the ancilla
    for q in range(num_qubits):
        H * qr[q]

    # infer the output basis state
    ind = 2 * secret + 1
    prob = circuit.get_prob(ind)
    if 1 - prob < PRECISION:
        prob = 1

    print(f"success probability:  {prob}")

    return circuit


if __name__ == "__main__":
    # number of qubits
    num_qubits = 15

    # number of all states
    num_elems = 2 ** (num_qubits - 1)

    # randomly choose the secret parameter
    secret = random.randint(0, num_elems - 1)

    # search for s using BV's algorithm
    circuit = apply_bernstein_vazirani(num_qubits, secret)

