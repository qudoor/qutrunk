import numpy as np
import pytest

from qutrunk.circuit.gates import H, All, X1
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Matrix
from qutrunk.test.gate.backend_fixture import backend, backend_type


class TestX1:
    @pytest.fixture
    def result_qutrunk(self, backend):
        # local backend
        circuit = QCircuit(backend=backend)
        qr = circuit.allocate(1)
        All(H) * qr
        X1 * qr[0]
        result_q = np.array(circuit.get_statevector()).reshape(-1, 1)
        return result_q

    def test_matrix(self, result_qutrunk):
        """Test X1 gate with Matrix."""
        circuit = QCircuit()
        qr = circuit.allocate(1)
        Matrix(H.matrix.tolist()) * qr[0]
        Matrix(X1.matrix.tolist()) * qr[0]
        result_m = circuit.get_statevector()
        result_m = np.array(result_m).reshape(-1, 1)
        assert np.allclose(result_qutrunk, result_m)

    def test_gate_inverse(self):
        """Test the inverse of X1 gate."""
        # local backend
        circuit = QCircuit()
        qr = circuit.allocate(1)
        All(H) * qr
        # initial state
        result_init = np.array(circuit.get_statevector()).reshape(-1, 1)

        X1 * qr[0]
        X1.inv() * qr[0]
        result_expect = circuit.get_statevector()
        result_expect = np.array(result_expect).reshape(-1, 1)

        assert np.allclose(result_init, result_expect)

    def test_matrix_inverse(self):
        """Test the inverse of X1 gate with Matrix."""
        circuit = QCircuit()
        qr = circuit.allocate(1)

        Matrix(H.matrix.tolist()) * qr[0]
        # initial state
        result_init = np.array(circuit.get_statevector()).reshape(-1, 1)

        Matrix(X1.matrix.tolist()) * qr[0]
        Matrix(X1.matrix.tolist()).inv() * qr[0]
        result_m = circuit.get_statevector()
        result_m = np.array(result_m).reshape(-1, 1)

        assert np.allclose(result_init, result_m)
