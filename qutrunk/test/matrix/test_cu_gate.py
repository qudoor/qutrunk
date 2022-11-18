import numpy as np
from numpy import pi

from qutrunk.circuit.gates import X, CU
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_cu_gate():
    """Test CU matrix."""
    # cu gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(2)
    X * qr[0]
    CU(pi / 2, pi / 2, pi / 2) * (qr[0], qr[1])
    result = circuit.get_statevector()
    result_cu = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(2)
    Matrix(X.matrix.tolist()) * _qr[0]
    Matrix(CU(pi / 2, pi / 2, pi / 2).matrix.tolist(), 1) * (_qr[0], _qr[1])
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_cu, result_matrix)


def test_cu_inverse_gate():
    """Test CU matrix inverse."""
    # cu gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(2)
    X * qr[0]
    CU(pi / 2, pi / 2, pi / 2).inv() * (qr[0], qr[1])
    result = circuit.get_statevector()
    result_cu = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(2)
    Matrix(X.matrix.tolist()) * _qr[0]
    Matrix(CU(pi / 2, pi / 2, pi / 2).matrix.tolist(), 1).inv() * (_qr[0], _qr[1])
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_cu, result_matrix)


