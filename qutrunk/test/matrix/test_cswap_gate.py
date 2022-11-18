import numpy as np

from qutrunk.circuit.gates import X, CSwap
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_cswap_gate():
    """Test CSwap matrix."""
    # cswap gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(3)
    X * qr[0]
    CSwap * (qr[0], qr[1], qr[2])
    result = circuit.get_statevector()
    result_cswap = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(3)
    Matrix(X.matrix.tolist()) * _qr[0]
    Matrix(CSwap.matrix.tolist(), 1) * (_qr[0], _qr[1], _qr[2])
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_cswap, result_matrix)


def test_cswap_inverse_gate():
    """Test CSwap matrix inverse."""
    # cswap gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(3)
    X * qr[0]
    CSwap.inv() * (qr[0], qr[1], qr[2])
    result = circuit.get_statevector()
    result_cswap = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(3)
    Matrix(X.matrix.tolist()) * _qr[0]
    Matrix(CSwap.matrix.tolist(), 1).inv() * (_qr[0], _qr[1], _qr[2])
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_cswap, result_matrix)