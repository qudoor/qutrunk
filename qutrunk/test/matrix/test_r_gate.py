import numpy as np
from numpy import pi

from qutrunk.circuit.gates import R
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_r_gate():
    """Test R gate."""
    # r gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    R(pi / 2, pi / 2) * qr[0]
    result = circuit.get_statevector()
    result_r = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(R(pi / 2, pi / 2).matrix.tolist()) * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_r, result_matrix)


def test_r_inverse_gate():
    """Test the inverse of R gate."""
    # r gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    R(pi / 2, pi / 2).inv() * qr[0]
    result = circuit.get_statevector()
    result_r = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(R(pi / 2, pi / 2).matrix.tolist()).inv() * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_r, result_matrix)

