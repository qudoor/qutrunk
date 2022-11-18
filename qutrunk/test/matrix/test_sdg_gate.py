import numpy as np

from qutrunk.circuit.gates import Sdg
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_sdg_gate():
    """Test Sdg gate."""
    # sdg gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    Sdg * qr[0]
    result = circuit.get_statevector()
    result_sdg = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(Sdg.matrix.tolist()) * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_sdg, result_matrix)


def test_sdg_inverse_gate():
    """Test the inverse of Sdg gate."""
    # sdg gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    Sdg.inv() * qr[0]
    result = circuit.get_statevector()
    result_sdg = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(Sdg.matrix.tolist()).inv() * _qr[0]
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_sdg, result_matrix)



