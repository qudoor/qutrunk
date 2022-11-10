import numpy as np

from qutrunk.circuit.gates import X, CZ
from qutrunk.circuit import QCircuit


def test_cz_gate():
    """Test CZ gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(2)

    X * qr[0]
    CZ * (qr[0], qr[1])
    CZ * (qr[0], qr[1])
    result_backend = circuit.get_statevector()

    # initial state
    result = np.array([0, 1, 0, 0])
    assert np.allclose(result_backend, result)
