import numpy as np

from qutrunk.circuit.gates import X, CSwap
from qutrunk.circuit import QCircuit


def test_csqrtx_gate():
    """Test CSwap gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(3)

    X * qr[0]
    CSwap * (qr[0], qr[1], qr[2])
    CSwap.inv() * (qr[0], qr[1], qr[2])
    result_backend = circuit.get_statevector()

    # initial state
    result = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    assert np.allclose(result_backend, result)
