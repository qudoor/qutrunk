import pytest

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import NOT
from qutrunk.backends import BackendQuSprout
from check_all_state import check_all_state
from check_all_state_inverse import check_all_state_inverse


import numpy as np

from qutrunk.circuit import QCircuit
from qutrunk.test.global_parameters import ZERO_STATE


def test_not_gate():
    """Test NOT gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    NOT * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)
    print(result_backend)

    # math
    result_math = np.dot(NOT.matrix, ZERO_STATE)
    print(result_math)

    assert np.allclose(result_backend, result_math)


def test_not_inverse_gate():
    """Test the inverse of NOT gate."""
    # local backend
    circuit = QCircuit()
    qr = circuit.allocate(1)
    NOT.inv() * qr[0]
    result = circuit.get_statevector()
    result_backend = np.array(result).reshape(-1, 1)
    print(result_backend)

    # math
    result_math = np.dot(NOT.matrix, ZERO_STATE)
    print(result_math)

    assert np.allclose(result_backend, result_math)
