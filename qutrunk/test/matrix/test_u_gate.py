import numpy as np
from numpy import pi

from qutrunk.circuit.gates import X, U
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_u_gate():
    """Test U gate."""
    # u gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    U(pi / 2, pi / 2, pi / 2) * (qr[0]) * qr[0]
    result = circuit.get_statevector()
    result_u = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(U(pi / 2, pi / 2, pi / 2).matrix.tolist()) * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_u, result_matrix)


def test_u_inverse_gate():
    """Test the inverse of U gate."""
    # u gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    U(pi / 2, pi / 2, pi / 2).inv() * qr[0]
    result = circuit.get_statevector()
    result_u = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(U(pi / 2, pi / 2, pi / 2).matrix.tolist()).inv() * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_u, result_matrix)
