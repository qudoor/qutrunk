import numpy as np

from qutrunk.circuit.gates import P
from qutrunk.circuit import QCircuit
from qutrunk.test.global_parameters import ZERO_STATE


def test_p_gate():
    """Test P gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    P(np.pi / 2) * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    p = P(np.pi / 2)
    result_math = np.dot(p.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)


def test_p_inverse_gate():
    """Test the inverse of P gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    P(np.pi / 2).inv() * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    p = P(-np.pi / 2)
    result_math = np.dot(p.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)
