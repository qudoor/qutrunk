import numpy as np

from qutrunk.circuit.gates import Z1
from qutrunk.circuit import QCircuit
from qutrunk.test.global_parameters import ZERO_STATE


def test_z1_gate():
    """Test Z1 gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    Z1 * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    result_math = np.dot(Z1.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)


def test_z1_inverse_gate():
    """Test the inverse of Z1 gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    Z1.inv() * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    a00 = complex(np.cos(np.pi / 4), np.sin(np.pi / 4))
    a11 = complex(np.cos(-np.pi / 4), np.sin(-np.pi / 4))
    m = np.array([[a00, 0], [0, a11]])
    result_math = np.dot(m, ZERO_STATE)

    assert np.allclose(result_backend, result_math)
