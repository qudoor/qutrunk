import numpy as np
from numpy import pi

from qutrunk.circuit.gates import U2
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_u2_gate():
    """Test U2 gate."""
    # u2 gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    U2(0, pi) * qr[0]
    result = circuit.get_statevector()
    result_u2 = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(U2(0, pi).matrix.tolist()) * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_u2, result_matrix)


def test_u2_inverse_gate():
    """Test the inverse of U2 gate."""
    # u2 gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    U2(0, pi).inv() * qr[0]
    result = circuit.get_statevector()
    result_u2 = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(U2(0, pi).matrix.tolist()).inv() * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_u2, result_matrix)
