import pytest
import json

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, S, X, U1, U2, U3
from numpy import pi


def test_u1_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    U1(pi / 2) | qr[0]
    # S | qr[0]
    target = '["real, imag", "1.000000000000, 0.000000000000", "0.000000000000, 0.000000000000"]'
    res = json.dumps(circuit.get_all_state())
    # print(circuit.get_all_state())
    assert res == target
    circuit.run()


def test_u2_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    U2(0, pi) | qr[0]
    # H | qr[0]
    target = '["real, imag", "0.707106781187, 0.000000000000", "0.707106781187, 0.000000000000"]'
    res = json.dumps(circuit.get_all_state())
    # print(circuit.get_all_state()
    assert res == target
    circuit.run()


def test_u3_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    U3(pi, 0, pi) | qr[0]
    # X | qr[0]
    target = '["real, imag", "0.000000000000, 0.000000000000", "1.000000000000, 0.000000000000"]'
    res = json.dumps(circuit.get_all_state())
    # print(circuit.get_all_state())
    assert res == target
    circuit.run()
