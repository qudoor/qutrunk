import numpy as np

from qutrunk.circuit.gates import X1
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_x1_gate():
    """Test X1 gate."""
    # x1 gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    X1 * qr[0]
    result = circuit.get_statevector()
    result_x1 = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(X1.matrix.tolist()) * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_x1, result_matrix)


def test_x1_inverse_gate():
    """Test the inverse of X1 gate."""
    # x1 gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    X1.inv() * qr[0]
    result = circuit.get_statevector()
    result_x1 = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(X1.matrix.tolist()).inv() * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_x1, result_matrix)
