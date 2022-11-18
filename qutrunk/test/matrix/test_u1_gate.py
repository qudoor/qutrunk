import numpy as np
from numpy import pi

from qutrunk.circuit.gates import U1
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_u1_gate():
    """Test U1 gate."""
    # u1 gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    U1(pi / 2) * qr[0]
    result = circuit.get_statevector()
    result_u1 = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(U1(pi / 2).matrix.tolist()) * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_u1, result_matrix)


def test_u1_inverse_gate():
    """Test the inverse of U1 gate."""
    # u1 gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    U1(pi / 2).inv() * qr[0]
    result = circuit.get_statevector()
    result_u1 = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(U1(pi / 2).matrix.tolist()).inv() * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_u1, result_matrix)


