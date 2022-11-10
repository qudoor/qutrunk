import numpy as np

from qutrunk.circuit.gates import CH, X
from qutrunk.circuit import QCircuit


def test_not_gate():
    """Test CH gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(2)
    X * qr[0]
    CH * (qr[0], qr[1])
    CH.inv() * (qr[0], qr[1])
    result_backend = circuit.get_statevector()

    # initial state
    result = np.array([0, 1, 0, 0])
    assert np.allclose(result_backend, result)

