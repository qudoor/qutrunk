import numpy as np
from numpy import pi

from qutrunk.circuit.gates import X, Rzz
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_rzz_gate():
    # rzz gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    X * qr[0]
    Rzz(pi / 2) * (qr[0], qr[1])
    result = circuit.get_statevector()
    result_rzz = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(X.matrix.tolist()) * _qr[0]
    Matrix(Rzz(pi / 2).matrix.tolist(), 1) * (qr[0], qr[1])
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_rzz, result_matrix)


def test_rzz_inverse_gate():
    # rzz gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    X * qr[0]
    Rzz(pi / 2).inv() * (qr[0], qr[1])
    result = circuit.get_statevector()
    result_rzz = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(X.matrix.tolist()) * _qr[0]
    Matrix(Rzz(pi / 2).matrix.tolist(), 1).inv() * (qr[0], qr[1])
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_rzz, result_matrix)

