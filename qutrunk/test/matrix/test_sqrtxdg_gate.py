import numpy as np

from qutrunk.circuit.gates import SqrtXdg
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_sqrtx_gate():
    """Test SqrtXdg gate."""
    # sqrtxdg gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    SqrtXdg * qr[0]
    result = circuit.get_statevector()
    result_sqrtxdg = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(SqrtXdg.matrix.tolist()) * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_sqrtxdg, result_matrix)


def test_sqrtx_inverse_gate():
    """Test the inverse of SqrtXdg gate."""
    # sqrtxdg gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    SqrtXdg.inv() * qr[0]
    result = circuit.get_statevector()
    result_sqrtxdg = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(SqrtXdg.matrix.tolist()).inv() * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_sqrtxdg, result_matrix)
