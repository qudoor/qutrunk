import numpy as np

from qutrunk.circuit.gates import T
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_t_gate():
    """Test T gate."""
    # t gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    T * qr[0]
    result = circuit.get_statevector()
    result_t = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(T.matrix.tolist()) * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_t, result_matrix)


def test_t_inverse_gate():
    """Test the inverse of T gate."""
    # t gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    T.inv() * qr[0]
    result = circuit.get_statevector()
    result_t = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(T.matrix.tolist()).inv() * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_t, result_matrix)
