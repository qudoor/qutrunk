import numpy as np

from qutrunk.circuit.gates import SqrtX
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_sqrtx_gate():
    """Test SqrtX gate."""
    # sqrtx gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    SqrtX * qr[0]
    result = circuit.get_statevector()
    result_sqrtx = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(SqrtX.matrix.tolist()) * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_sqrtx, result_matrix)


def test_sqrtx_inverse_gate():
    """Test the inverse of SqrtX gate."""
    # sqrtx gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    SqrtX.inv() * qr[0]
    result = circuit.get_statevector()
    result_sqrtx = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(SqrtX.matrix.tolist()).inv() * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_sqrtx, result_matrix)
