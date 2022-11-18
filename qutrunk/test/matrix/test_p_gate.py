import numpy as np
from numpy import pi

from qutrunk.circuit.gates import P
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_p_gate():
    """Test P gate."""
    # p gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    P(pi / 2) * qr[0]
    result = circuit.get_statevector()
    result_p = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(P(pi / 2).matrix.tolist()) * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_p, result_matrix)


def test_p_inverse_gate():
    """Test the inverse of P gate."""
    # p gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    P(pi / 2).inv() * qr[0]
    result = circuit.get_statevector()
    result_p = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(P(pi / 2).matrix.tolist()).inv() * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_p, result_matrix)
