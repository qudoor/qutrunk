import numpy as np

from qutrunk.circuit.gates import Z
from qutrunk.circuit import QCircuit
from qutrunk.test.global_parameters import ZERO_STATE


def test_z_gate():
    """Test Z gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    Z * qr[0]
    result = circuit.get_statevector()

    result_backend = np.array(result).reshape(-1, 1)

    # math
    result_math = np.dot(Z.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)


def test_z_inverse_gate():
    """Test the inverse of Z gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    Z.inv() * qr[0]
    result = circuit.get_statevector()

    result_backend = np.array(result).reshape(-1, 1)

    # math
    result_math = np.dot(Z.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)
