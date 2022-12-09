import numpy as np
import pytest

from qutrunk.circuit.gates import H, All, Z1
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Matrix
from qutrunk.test.gate.backend_fixture import backend, backend_type


class TestZ1:
    @pytest.fixture
    def result_qutrunk(self, backend):
        # local backend
        circuit = QCircuit(backend=backend)
        qr = circuit.allocate(1)
        All(H) * qr
        Z1 * qr[0]
        result_q = np.array(circuit.get_statevector()).reshape(-1, 1)
        return result_q

    def test_matrix(self, result_qutrunk):
        """Test Z1 gate with Matrix."""
        circuit = QCircuit()
        qr = circuit.allocate(1)
        Matrix(H.matrix.tolist()) * qr[0]
        Matrix(Z1.matrix.tolist()) * qr[0]
        result_m = circuit.get_statevector()
        result_m = np.array(result_m).reshape(-1, 1)
        assert np.allclose(result_qutrunk, result_m)

    def test_gate_inverse(self):
        """Test the inverse of Z1 gate."""
        # local backend
        circuit = QCircuit()
        qr = circuit.allocate(1)
        All(H) * qr
        # initial state
        result_init = np.array(circuit.get_statevector()).reshape(-1, 1)

        Z1 * qr[0]
        Z1.inv() * qr[0]
        result_expect = circuit.get_statevector()
        result_expect = np.array(result_expect).reshape(-1, 1)

        assert np.allclose(result_init, result_expect)

    def test_matrix_inverse(self):
        """Test the inverse of Z1 gate with Matrix."""
        circuit = QCircuit()
        qr = circuit.allocate(1)

        Matrix(H.matrix.tolist()) * qr[0]
        # initial state
        result_init = np.array(circuit.get_statevector()).reshape(-1, 1)

        Matrix(Z1.matrix.tolist()) * qr[0]
        Matrix(Z1.matrix.tolist()).inv() * qr[0]
        result_m = circuit.get_statevector()
        result_m = np.array(result_m).reshape(-1, 1)

        assert np.allclose(result_init, result_m)
