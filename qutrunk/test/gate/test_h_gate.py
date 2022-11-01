import numpy as np

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H
from qutrunk.test.global_parameters import ZERO_STATE


def test_h_gate():
    """Test H gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    H * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    result_math = np.dot(H.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)


def test_h_inverse_gate():
    """Test the inverse of H gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    H.inv() * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    result_math = np.dot(H.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)

