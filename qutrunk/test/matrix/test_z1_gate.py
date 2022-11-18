import numpy as np

from qutrunk.circuit.gates import Z1
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_z1_gate():
    """Test Z1 gate."""
    # z1 gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    Z1 * qr[0]
    result = circuit.get_statevector()
    result_z1 = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(Z1.matrix.tolist()) * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_z1, result_matrix)


def test_z1_inverse_gate():
    """Test the inverse of Z1 gate."""
    # z1 gate inverse
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    Z1.inv() * qr[0]
    result = circuit.get_statevector()
    result_z1 = np.array(result).reshape(-1, 1)

    # matrix gate inverse
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(Z1.matrix.tolist()).inv() * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_z1, result_matrix)