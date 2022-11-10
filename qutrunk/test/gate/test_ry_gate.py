import numpy as np

from qutrunk.circuit.gates import X, Ry
from qutrunk.circuit import QCircuit


def test_ry_gate():
    """Test Ry gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(2)
    pi = np.pi

    X * qr[0]
    Ry(pi / 2) * qr[0]
    Ry(pi / 2).inv() * qr[0]
    result_backend = circuit.get_statevector()

    # initial state
    result = np.array([0, 1, 0, 0])
    assert np.allclose(result_backend, result)
