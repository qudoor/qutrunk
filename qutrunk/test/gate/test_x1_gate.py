import numpy as np

from qutrunk.circuit.gates import X1
from qutrunk.circuit import QCircuit
from qutrunk.test.global_parameters import ZERO_STATE


def test_x1_gate():
    """Test X1 gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    X1 * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    result_math = np.dot(X1.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)


def test_x1_inverse_gate():
    """Test the inverse of X1 gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    X1.inv() * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    factor = np.sqrt(2)
    a00 = 0.5 * factor
    a01 = complex(0, 0.5 * factor)
    m = np.array([[a00, a01], [a01, a00]])
    result_math = np.dot(m, ZERO_STATE)

    assert np.allclose(result_backend, result_math)
