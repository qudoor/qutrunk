import numpy as np
from numpy import pi
from scipy.optimize import minimize

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Ry, Rx, CNOT, Rz, X, Z, Y, All

np.set_printoptions(precision=4, suppress=True)

# See DOI: 10.1103/PhysRevX.6.031007
# Here, we use parameters given for H2 at R=0.75A
g0 = -0.4804
g1 = +0.3435
g2 = -0.4347
g3 = +0.5716
g4 = +0.0910
g5 = +0.0910

nuclear_repulsion = 0.7055696146


def ansatz(theta, qreg):
    Ry(pi / 2) * qreg[1]
    Rx(pi / 2) * qreg[0]  # todo minus required

    CNOT * (qreg[1], qreg[0])

    Rz(theta) * qreg[0]

    CNOT * (qreg[1], qreg[0])

    Ry(pi / 2) * qreg[1]  # todo minus required
    Rx(pi / 2) * qreg[0]


def projective_expected(a_theta, a_ansatz):
    energy = g0

    circuit, qreg = prepare(a_theta, a_ansatz)
    in_state = circuit.get_statevector()

    Z * qreg[0]
    out_state = circuit.get_statevector()
    energy += g1 * state_inner(in_state, out_state)
    Z * qreg[0]  # cancel op

    Z * qreg[1]
    out_state = circuit.get_statevector()
    energy += g2 * state_inner(in_state, out_state)
    Z * qreg[1]

    All(Z) * qreg
    out_state = circuit.get_statevector()
    energy += g3 * state_inner(in_state, out_state)
    All(Z) * qreg

    All(X) * qreg
    out_state = circuit.get_statevector()
    energy += g4 * state_inner(in_state, out_state)
    All(X) * qreg

    All(Y) * qreg
    out_state = circuit.get_statevector()
    energy += g5 * state_inner(in_state, out_state)
    All(Y) * qreg

    # op_list = [3, 0, 0, 3, 3, 3, 1, 1, 2, 2]
    # coff_list = [g1, g2, g3, g4, g5]
    # res += circuit.expval_sum([op_list, coff_list], 2)
    return energy


def prepare(a_theta, a_ansatz):
    circuit = QCircuit()
    qreg = circuit.allocate(2)
    X * qreg[0]

    a_ansatz(a_theta[0], qreg)
    return circuit, qreg


def state_inner(i_state, o_state):
    res = i_state @ o_state.conj().T
    return res.real


theta = [0.0]
result = minimize(projective_expected, theta, args=ansatz)
theta = result.x[0]
val = result.fun

# check it works...
# assert np.allclose(val + nuclear_repulsion,-1.1456295)

print("VQE: ")
print("  [+] theta:  {:+2.8} deg".format(theta))
print("  [+] energy: {:+2.8} Eh".format(val + nuclear_repulsion))
