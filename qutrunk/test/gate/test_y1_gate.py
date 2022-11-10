import numpy as np

from qutrunk.circuit.gates import Y1
from qutrunk.circuit import QCircuit
from qutrunk.test.global_parameters import ZERO_STATE


def test_y1_gate():
    """Test Y1 gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    Y1 * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    result_math = np.dot(Y1.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)


def test_y1_inverse_gate():
    """Test the inverse of Y1 gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    Y1.inv() * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    factor = np.sqrt(2)
    a = 0.5 * factor
    m = np.array([[a, a], [-a, a]])
    result_math = np.dot(m, ZERO_STATE)

    assert np.allclose(result_backend, result_math)

