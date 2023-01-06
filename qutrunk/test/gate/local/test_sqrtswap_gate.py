import numpy as np
import pytest
from qiskit import QuantumCircuit, assemble, Aer

from qutrunk.circuit.gates import H, All, SqrtSwap
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Matrix
from qutrunk.test.gate.local.backend_fixture import backend, backend_type


class TestSqrtSwap:
    @pytest.fixture
    def result_qutrunk(self, backend):
        # local backend
        circuit = QCircuit(backend=backend)
        qr = circuit.allocate(2)
        All(H) * qr
        SqrtSwap * (qr[0], qr[1])
        result_q = np.array(circuit.get_statevector()).reshape(-1, 1)
        return result_q

    def test_matrix(self, result_qutrunk):
        """Test SqrtSwap gate with Matrix."""
        circuit = QCircuit()
        qr = circuit.allocate(2)
        Matrix(H.matrix.tolist()) * qr[0]
        Matrix(H.matrix.tolist()) * qr[1]
        Matrix(SqrtSwap.matrix.tolist()) * (qr[0], qr[1])
        result_m = circuit.get_statevector()
        result_m = np.array(result_m).reshape(-1, 1)

        assert np.allclose(result_qutrunk, result_m)

    def test_gate_inverse(self):
        """Test the inverse of SqrtSwap gate."""
        # local backend
        circuit = QCircuit()
        qr = circuit.allocate(2)
        All(H) * qr
        # initial state
        result_init = np.array(circuit.get_statevector()).reshape(-1, 1)

        SqrtSwap * (qr[0], qr[1])
        SqrtSwap.inv() * (qr[0], qr[1])
        result_expect = circuit.get_statevector()
        result_expect = np.array(result_expect).reshape(-1, 1)

        assert np.allclose(result_init, result_expect)

    def test_matrix_inverse(self):
        """Test the inverse of SqrtSwap gate with Matrix."""
        circuit = QCircuit()
        qr = circuit.allocate(2)

        Matrix(H.matrix.tolist()) * qr[0]
        Matrix(H.matrix.tolist()) * qr[1]
        # initial state
        result_init = np.array(circuit.get_statevector()).reshape(-1, 1)

        Matrix(SqrtSwap.matrix.tolist()) * (qr[0], qr[1])
        Matrix(SqrtSwap.matrix.tolist()).inv() * (qr[0], qr[1])
        result_m = circuit.get_statevector()
        result_m = np.array(result_m).reshape(-1, 1)

        assert np.allclose(result_init, result_m)
