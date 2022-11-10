import numpy as np

from qutrunk.circuit.gates import U1
from qutrunk.circuit import QCircuit
from qutrunk.test.global_parameters import ZERO_STATE


def test_u1_gate():
    """Test U1 gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    U1(np.pi / 2) * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    u1 = U1(np.pi / 2)
    result_math = np.dot(u1.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)


def test_u1_inverse_gate():
    """Test the inverse of U1 gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    U1(np.pi / 2).inv() * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    m = np.array([
        [1, np.cos(-np.pi/2)],
        [0, np.sin(-np.pi/2)]
    ])
    result_math = np.dot(m, ZERO_STATE)

    assert np.allclose(result_backend, result_math)

