import numpy as np

from qutrunk.circuit.gates import NOT
from qutrunk.circuit import QCircuit
from qutrunk.test.global_parameters import ZERO_STATE


def test_not_gate():
    """Test NOT gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    NOT * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    result_math = np.dot(NOT.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)


def test_not_inverse_gate():
    """Test the inverse of NOT gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    NOT.inv() * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    result_math = np.dot(NOT.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)
