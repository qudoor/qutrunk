import numpy as np

from qutrunk.circuit.gates import Z
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_z_gate():
    """Test Z gate."""
    # z gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    Z * qr[0]
    result = circuit.get_statevector()
    result_z = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(Z.matrix.tolist()) * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_z, result_matrix)


def test_z_inverse_gate():
    """Test the inverse of Z gate."""
    # z gate inverse
    circuit = QCircuit()
    qr = circuit.allocate(1)
    Z.inv() * qr[0]
    result = circuit.get_statevector()
    result_z = np.array(result).reshape(-1, 1)

    # matrix gate inverse
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(Z.matrix.tolist()).inv() * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_z, result_matrix)
