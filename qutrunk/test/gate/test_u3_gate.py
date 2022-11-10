import numpy as np
from numpy import pi

from qutrunk.circuit.gates import U3
from qutrunk.circuit import QCircuit
from qutrunk.test.global_parameters import ZERO_STATE


def test_u3_gate():
    """Test U3 gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    U3(pi, 0, pi) * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    u3 = U3(pi, 0, pi)
    result_math = np.dot(u3.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)


def test_u3_inverse_gate():
    """Test the inverse of U3 gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    U3(pi, 0, pi).inv() * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    theta = -pi
    phi = -pi
    lam = 0

    a00 = np.cos(theta/2)
    a01 = complex(-1 * np.cos(lam) * np.sin(theta / 2), -1 * np.sin(lam) * np.sin(theta / 2))
    a10 = complex(np.cos(phi) * np.sin(theta / 2), np.sin(phi) * np.sin(theta / 2))
    a11 = complex(np.cos(phi + lam) * np.cos(theta / 2), np.sin(phi + lam) * np.cos(theta / 2))
    m = np.array([
        [a00, a01],
        [a10, a11],
    ])
    result_math = np.dot(m, ZERO_STATE)

    assert np.allclose(result_backend, result_math)

