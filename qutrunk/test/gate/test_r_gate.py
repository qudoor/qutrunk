import numpy as np
from numpy import pi

from qutrunk.circuit.gates import R
from qutrunk.circuit import QCircuit
from qutrunk.test.global_parameters import ZERO_STATE


def test_r_gate():
    """Test R gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    R(pi / 2, pi / 2) * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    r = R(pi / 2, pi / 2)
    result_math = np.dot(r.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)


def test_r_inverse_gate():
    """Test the inverse of R gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    R(pi / 2, pi / 2).inv() * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    theta = -pi / 2
    phi = pi / 2

    a00 = np.cos(theta / 2)
    a01 = complex(np.sin(-phi) * np.sin(theta / 2), -1 * np.cos(-phi) * np.sin(theta / 2))
    a10 = complex(np.sin(phi) * np.sin(theta / 2), -1 * np.cos(phi) * np.sin(theta / 2))
    a11 = np.cos(theta / 2)
    m = np.array([
        [a00, a01],
        [a10, a11],
    ])
    result_math = np.dot(m, ZERO_STATE)

    assert np.allclose(result_backend, result_math)
