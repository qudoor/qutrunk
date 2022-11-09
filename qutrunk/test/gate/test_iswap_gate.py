import numpy as np

from qutrunk.circuit.gates import X, iSwap
from qutrunk.circuit import QCircuit


def test_iswap_gate():
    """Test iSwap gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(2)

    X * qr[0]
    iSwap * (qr[0], qr[1])
    iSwap.inv() * (qr[0], qr[1])
    result_backend = circuit.get_statevector()

    # initial state
    result = np.array([0, 1, 0, 0])
    assert np.allclose(result_backend, result)
