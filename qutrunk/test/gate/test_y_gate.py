import numpy as np


from qutrunk.circuit.gates import Y
from qutrunk.circuit import QCircuit
from qutrunk.test.global_parameters import ZERO_STATE


def test_y_gate():
    """Test Y gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    Y * qr[0]
    result = circuit.get_statevector()

    result_backend = np.array(result).reshape(-1, 1)

    # math
    result_math = np.dot(Y.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)


def test_y_inverse_gate():
    """Test the inverse of Y gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    Y.inv() * qr[0]
    result = circuit.get_statevector()

    result_backend = np.array(result).reshape(-1, 1)

    # math
    result_math = np.dot(Y.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)

