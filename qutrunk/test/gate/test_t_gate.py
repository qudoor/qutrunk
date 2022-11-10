import numpy as np

from qutrunk.circuit.gates import T, Tdg
from qutrunk.circuit import QCircuit
from qutrunk.test.global_parameters import ZERO_STATE


def test_t_gate():
    """Test T gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    T * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    result_math = np.dot(Tdg.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)


def test_t_inverse_gate():
    """Test the inverse of T gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    T.inv() * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)

    # math
    result_math = np.dot(Tdg.matrix, ZERO_STATE)

    assert np.allclose(result_backend, result_math)
