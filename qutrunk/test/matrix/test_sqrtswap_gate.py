import numpy as np

from qutrunk.circuit.gates import X, SqrtSwap
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_sqrtswap_gate():
    """Test SqrtSwap gate."""
    # sqrtswap gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    X * qr[0]
    SqrtSwap * (qr[0], qr[1])
    result = circuit.get_statevector()
    result_sqrtswap = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(X.matrix.tolist()) * _qr[0]
    Matrix(SqrtSwap.matrix.tolist(), 1) * (qr[0], qr[1])
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_sqrtswap, result_matrix)


def test_sqrtswap_inverse_gate():
    """Test SqrtSwap inverse gate."""
    # sqrtswap gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    X * qr[0]
    SqrtSwap.inv() * (qr[0], qr[1])
    result = circuit.get_statevector()
    result_sqrtswap = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(X.matrix.tolist()) * _qr[0]
    Matrix(SqrtSwap.matrix.tolist(), 1).inv() * (qr[0], qr[1])
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_sqrtswap, result_matrix)
