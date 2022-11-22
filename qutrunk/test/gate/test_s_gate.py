import numpy as np
from qiskit import QuantumCircuit, assemble, Aer

from qutrunk.circuit.gates import S, Sdg
from qutrunk.circuit import QCircuit
from qutrunk.test.global_parameters import ZERO_STATE
from qutrunk.circuit.gates import Matrix
from qutrunk.backends import BackendQuSprout


class TestSgate:
    def test_s_gate_matrix(self):
        """Test S gate."""
        # local backend
        circuit_l = QCircuit()
        qr = circuit_l.allocate(1)
        S * qr[0]
        result_l = np.array(circuit_l.get_statevector()).reshape(-1, 1)

        # TODO: Matrix have some problem.
        # Matrix
        circuit_m = QCircuit()
        qr_m = circuit_m.allocate(1)
        # m = [[1, 0], [0, 1j]]
        Matrix(S.matrix.tolist()) * qr_m[0]
        result_m = circuit_m.get_statevector()
        result_m = np.array(result_m).reshape(-1, 1)

        assert np.allclose(result_l, result_m)

    def test_s_gate_qiskit(self):
        # local backend
        circuit_l = QCircuit()
        qr = circuit_l.allocate(1)
        S * qr[0]
        result = circuit_l.get_statevector()
        result_l = np.array(result).reshape(-1, 1)

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

        assert np.allclose(result_l, result_qiskit)

    # def test_s_inverse_gate(self):
    #     """Test the inverse of S gate."""
    #     # local backend
    #     circuit = QCircuit()
    #     qr = circuit.allocate(1)
    #     S.inv() * qr[0]
    #     result = circuit.get_statevector()
    #     result_backend = np.array(result).reshape(-1, 1)
    #
    #     # math
    #     result_math = np.dot(Sdg.matrix, ZERO_STATE)
    #
    #     assert np.allclose(result_backend, result_math)