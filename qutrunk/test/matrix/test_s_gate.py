import numpy as np

from qutrunk.circuit.gates import S
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_s_gate():
    """Test S gate."""
    # s gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    S * qr[0]
    result = circuit.get_statevector()
    result_s = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(S.matrix.tolist()) * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_s, result_matrix)


def test_s_inverse_gate():
    """Test the inverse of S gate."""
    # s gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    S.inv() * qr[0]
    result = circuit.get_statevector()
    result_s = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(S.matrix.tolist()).inv() * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_s, result_matrix)


