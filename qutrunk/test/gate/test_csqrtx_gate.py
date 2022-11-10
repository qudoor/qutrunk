import numpy as np

from qutrunk.circuit.gates import X, CSqrtX
from qutrunk.circuit import QCircuit


def test_csqrtx_gate():
    """Test CSqrtX gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(2)

    X * qr[0]
    CSqrtX * (qr[0], qr[1])
    CSqrtX.inv() * (qr[0], qr[1])
    result_backend = circuit.get_statevector()

    # initial state
    result = np.array([0, 1, 0, 0])
    assert np.allclose(result_backend, result)
