import numpy as np

from qutrunk.circuit.gates import S, Sdg
from qutrunk.circuit import QCircuit
from qutrunk.test.global_parameters import ZERO_STATE


def test_s_gate():
    """Test S gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    S * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    result_math = np.dot(S.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)


def test_s_inverse_gate():
    """Test the inverse of S gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    S.inv() * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    result_math = np.dot(Sdg.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)


