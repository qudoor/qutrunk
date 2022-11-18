import numpy as np

from qutrunk.circuit.gates import Y
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_y_gate():
    """Test Y gate."""
    # y gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    Y * qr[0]
    result = circuit.get_statevector()
    result_y = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(Y.matrix.tolist()) * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_y, result_matrix)


def test_y_inverse_gate():
    """Test the inverse of Y gate."""
    # y gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    Y.inv() * qr[0]
    result = circuit.get_statevector()
    result_y = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(Y.matrix.tolist()).inv() * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_y, result_matrix)

