import numpy as np

from qutrunk.circuit.gates import Y1
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_y1_gate():
    """Test Y1 gate."""
    # y1 gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    Y1 * qr[0]
    result = circuit.get_statevector()
    result_y1 = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(Y1.matrix.tolist()) * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_y1, result_matrix)


def test_y1_inverse_gate():
    """Test the inverse of Y1 gate."""
    # y1 gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    Y1.inv() * qr[0]
    result = circuit.get_statevector()
    result_y1 = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(Y1.matrix.tolist()).inv() * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_y1, result_matrix)

