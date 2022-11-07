import numpy as np

from qutrunk.circuit.gates import CX
from qutrunk.circuit import QCircuit
from qutrunk.test.global_parameters import ZERO_STATE_2D


def test_not_gate():
    """Test CX gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(2)
    CX * (qr[0], qr[1])
    CX * (qr[0], qr[1])
    result = circuit.get_statevector()
    result_backend = np.array(result)

    # initial state
    assert np.allclose(result_backend, ZERO_STATE_2D)
