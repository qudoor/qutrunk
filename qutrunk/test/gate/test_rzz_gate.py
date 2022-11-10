import numpy as np

from qutrunk.circuit.gates import X, Rzz
from qutrunk.circuit import QCircuit


def test_rzz_gate():
    """Test Rzz gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(2)
    pi = np.pi

    X * qr[0]
    Rzz(pi / 2) * (qr[0], qr[1])
    Rzz(pi / 2).inv() * (qr[0], qr[1])
    result_backend = circuit.get_statevector()

    # initial state
    result = np.array([0, 1, 0, 0])
    assert np.allclose(result_backend, result)
