import numpy as np

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_h_gate():
    """Test H matrix."""
    # h gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    H * qr[0]
    result = circuit.get_statevector()
    result_h = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(H.matrix.tolist()) * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_h, result_matrix)


def test_h_inverse_gate():
    """Test H matrix inverse."""
    # h gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    H.inv() * qr[0]
    result = circuit.get_statevector()
    result_h = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(H.matrix.tolist()).inv() * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_h, result_matrix)

