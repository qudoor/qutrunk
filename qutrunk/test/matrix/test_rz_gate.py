import numpy as np
from numpy import pi

from qutrunk.circuit.gates import X, Rz
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_rz_gate():
    """Test Rz gate."""
    # rz gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    X * qr[0]
    Rz(pi / 2) * qr[0]
    result = circuit.get_statevector()
    result_rz = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(X.matrix.tolist()) * _qr[0]
    Matrix(Rz(pi / 2).matrix.tolist()) * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_rz, result_matrix)


def test_rz_inverse_gate():
    """Test Rz gate."""
    # rz gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    X * qr[0]
    Rz(pi / 2).inv() * qr[0]
    result = circuit.get_statevector()
    result_rz = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(X.matrix.tolist()) * _qr[0]
    Matrix(Rz(pi / 2).matrix.tolist()).inv() * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_rz, result_matrix)

