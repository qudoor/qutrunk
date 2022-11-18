import numpy as np
from numpy import pi

from qutrunk.circuit.gates import X, Ryy
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

def test_ryy_gate():
    """Test Ryy gate."""
    # ryy gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    X * qr[0]
    Ryy(pi / 2) * (qr[0], qr[1])
    result = circuit.get_statevector()
    result_ryy = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(X.matrix.tolist()) * _qr[0]
    Matrix(Ryy(pi / 2).matrix.tolist(), 1) * (qr[0], qr[1])
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_ryy, result_matrix)


def test_ryy_inverse_gate():
    """Test Ryy inverse gate."""
    # ryy gate
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(1)
    X * qr[0]
    Ryy(pi / 2) * (qr[0], qr[1])
    result = circuit.get_statevector()
    result_ryy = np.array(result).reshape(-1, 1)

    # matrix gate
    _circuit = QCircuit(backend=BackendQuSprout())
    _qr = _circuit.allocate(1)
    Matrix(X.matrix.tolist()) * _qr[0]
    Matrix(Ryy(pi / 2).matrix.tolist(), 1) * (qr[0], qr[1])
    result = _circuit.get_statevector()
    result_matrix = np.array(result).reshape(-1, 1)

    assert np.allclose(result_ryy, result_matrix)
