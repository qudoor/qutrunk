import numpy as np

from qutrunk.circuit.gates import U2
from qutrunk.circuit import QCircuit
from qutrunk.test.global_parameters import ZERO_STATE


def test_u2_gate():
    """Test U2 gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    U2(0, np.pi) * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    u2 = U2(0, np.pi)
    result_math = np.dot(u2.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)


def test_u2_inverse_gate():
    """Test the inverse of U2 gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    U2(0, np.pi).inv() * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    phi = -2*np.pi
    lam = np.pi
    factor = 1 / np.sqrt(2)

    m = np.array([
        [1 * factor, complex(-factor * np.cos(lam), -factor * np.sin(lam))],
        [complex(factor * np.cos(phi), factor * np.sin(phi)),  complex(factor * np.cos(phi + lam), factor * np.sin(phi + lam) )]
    ])
    result_math = np.dot(m, ZERO_STATE)

    assert np.allclose(result_backend, result_math)
