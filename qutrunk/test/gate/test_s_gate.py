import numpy as np
import pytest
from qiskit import QuantumCircuit, assemble, Aer

from qutrunk.circuit.gates import S
from qutrunk.circuit import QCircuit
from qutrunk.test.global_parameters import ZERO_STATE
from qutrunk.circuit.gates import Matrix


class TestSgate:
    @pytest.fixture
    def result_local(self):
        # local backend
        circuit_l = QCircuit()
        qr = circuit_l.allocate(1)
        S * qr[0]
        result_l = np.array(circuit_l.get_statevector()).reshape(-1, 1)
        return result_l

    def test_matrix(self, result_local):
        """Test S gate with Matrix."""
        # Matrix
        circuit_m = QCircuit()
        qr_m = circuit_m.allocate(1)
        # m = [[1, 0], [0, 1j]]
        Matrix(S.matrix.tolist()) * qr_m[0]
        result_m = circuit_m.get_statevector()
        result_m = np.array(result_m).reshape(-1, 1)

        assert np.allclose(result_local, result_m)

    def test_qiskit(self, result_local):
        """Test S gate with qiskit."""
        # qiskit
        qc = QuantumCircuit(1)
        initial_state = [1, 0]
        qc.initialize(initial_state, 0)
        # apply gate
        qc.s(0)
        # run
        sim = Aer.get_backend("aer_simulator")
        qc.save_statevector()
        q_obj = assemble(qc)
        result = sim.run(q_obj).result()
        result_qiskit = np.array(result.get_statevector()).reshape(-1, 1)

        assert np.allclose(result_local, result_qiskit)

    def test_inverse_local(self):
        """Test the inverse of S gate."""
        # local backend
        circuit = QCircuit()
        qr = circuit.allocate(1)
        S * qr[0]
        S.inv() * qr[0]
        result_local = circuit.get_statevector()
        result_local = np.array(result_local).reshape(-1, 1)

        # initial state
        assert np.allclose(result_local, ZERO_STATE)

    def test_inverse_matrix(self):
        # matrix
        circuit_m = QCircuit()
        qr_m = circuit_m.allocate(1)

        Matrix(S.matrix.tolist()) * qr_m[0]
        Matrix(S.matrix.tolist()).inv() * qr_m[0]
        result_m = circuit_m.get_statevector()
        result_m = np.array(result_m).reshape(-1, 1)

        assert np.allclose(result_m, ZERO_STATE)
