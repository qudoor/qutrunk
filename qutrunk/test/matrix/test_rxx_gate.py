import numpy as np
from numpy import pi

from qutrunk.circuit.gates import X, Rxx
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_rxx_gate():
    """Test Rxx gate."""
    # rxx gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    X * qr[0]
    Rxx(pi / 2) * (qr[0], qr[1])
    result = circuit.get_statevector()
    result_rxx = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(X.matrix.tolist()) * _qr[0]
    Matrix(Rxx(pi / 2).matrix.tolist(), 1) * (qr[0], qr[1])
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_rxx, result_matrix)


def test_rxx_inverse_gate():
    """Test Rxx gate."""
    # rxx gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    X * qr[0]
    Rxx(pi / 2).inv() * (qr[0], qr[1])
    result = circuit.get_statevector()
    result_rxx = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(X.matrix.tolist()) * _qr[0]
    Matrix(Rxx(pi / 2).matrix.tolist(), 1).inv() * (qr[0], qr[1])
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_rxx, result_matrix)

