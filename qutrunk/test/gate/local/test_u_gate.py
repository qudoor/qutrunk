import numpy as np
import pytest
from qiskit import QuantumCircuit, assemble, Aer

from qutrunk.circuit.gates import H, All, U
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Matrix
from qutrunk.test.gate.local.backend_fixture import backend, backend_type


class TestU:
    @pytest.fixture
    def result_qutrunk(self, backend):
        # local backend
        circuit = QCircuit(backend=backend)
        qr = circuit.allocate(1)
        All(H) * qr
        U(np.pi / 2, np.pi / 2, np.pi / 2) * qr[0]
        result_q = np.array(circuit.get_statevector()).reshape(-1, 1)
        return result_q

    def test_matrix(self, result_qutrunk):
        """Test U gate with Matrix."""
        circuit = QCircuit()
        qr = circuit.allocate(1)
        Matrix(H.matrix.tolist()) * qr[0]
        Matrix(U(np.pi / 2, np.pi / 2, np.pi / 2).matrix.tolist()) * qr[0]
        result_m = circuit.get_statevector()
        result_m = np.array(result_m).reshape(-1, 1)
        assert np.allclose(result_qutrunk, result_m)

    def test_qiskit(self, result_qutrunk):
        """Test U gate with qiskit."""
        qc = QuantumCircuit(1)
        initial_state = [1, 0]
        qc.initialize(initial_state, 0)
        # apply gate
        qc.h(0)
        qc.u(np.pi / 2, np.pi / 2, np.pi / 2, 0)
        # run
        sim = Aer.get_backend("aer_simulator")
        qc.save_statevector()
        q_obj = assemble(qc)
        result_qiskit = sim.run(q_obj).result()
        result_qiskit = np.array(result_qiskit.get_statevector()).reshape(-1, 1)

        assert np.allclose(result_qutrunk, result_qiskit)

    def test_gate_inverse(self):
        """Test the inverse of U gate."""
        # local backend
        circuit = QCircuit()
        qr = circuit.allocate(1)
        All(H) * qr
        # initial state
        result_init = np.array(circuit.get_statevector()).reshape(-1, 1)

        U(np.pi / 2, np.pi / 2, np.pi / 2) * qr[0]
        U(np.pi / 2, np.pi / 2, np.pi / 2).inv() * qr[0]
        result_expect = circuit.get_statevector()
        result_expect = np.array(result_expect).reshape(-1, 1)

        assert np.allclose(result_init, result_expect)

    def test_matrix_inverse(self):
        """Test the inverse of U gate with Matrix."""
        circuit = QCircuit()
        qr = circuit.allocate(1)

        Matrix(H.matrix.tolist()) * qr[0]
        # initial state
        result_init = np.array(circuit.get_statevector()).reshape(-1, 1)

        Matrix(U(np.pi / 2, np.pi / 2, np.pi / 2).matrix.tolist()) * qr[0]
        Matrix(U(np.pi / 2, np.pi / 2, np.pi / 2).matrix.tolist()).inv() * qr[0]
        result_m = circuit.get_statevector()
        result_m = np.array(result_m).reshape(-1, 1)

        assert np.allclose(result_init, result_m)
