import pytest
import numpy as np
from numpy import pi

from qiskit import QuantumCircuit, BasicAer, transpile

from qutrunk.circuit.gates import R, All, H
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Matrix
from qutrunk.test.gate.backend_fixture import backend, backend_type

class Test_R_Gate:
    @pytest.fixture
    def result_gate(self, backend):
        circuit = QCircuit(backend=backend)
        qr = circuit.allocate(1)
        All(H) * qr
        R(pi / 2, pi / 2) * qr[0]
        result_gate = np.array(circuit.get_statevector()).reshape(-1, 1)
        return result_gate

    def test_result_matrix(self, result_gate):
        circuit = QCircuit()
        qr = circuit.allocate(1)
        Matrix(H.matrix.tolist()) * qr[0]
        Matrix(R(pi / 2, pi / 2).matrix.tolist()) * qr[0]
        result_matrix = np.array(circuit.get_statevector()).reshape(-1, 1)
        assert np.allclose(result_gate, result_matrix)

    def test_result_qiskit(self, result_gate):
        qc = QuantumCircuit(1, 1)
        backend = BasicAer.get_backend('statevector_simulator')
        qc.h(0)
        qc.r(pi / 2, pi / 2,0)
        job = backend.run(transpile(qc, backend))
        result_qiskit = np.array(job.result().get_statevector(qc)).reshape(-1, 1)
        assert np.allclose(result_gate, result_qiskit)

    def test_gate_inverse(self, backend):
        circuit = QCircuit(backend=backend)
        qr = circuit.allocate(1)
        All(H) * qr
        result_src = np.array(circuit.get_statevector()).reshape(-1, 1)
        R(pi / 2, pi / 2) * qr[0]
        R(pi / 2, pi / 2).inv() * qr[0]
        result_des = np.array(circuit.get_statevector()).reshape(-1, 1)
        assert np.allclose(result_src, result_des)

    def test_matrix_inverse(self, backend):
        circuit = QCircuit(backend=backend)
        qr = circuit.allocate(1)
        Matrix(H.matrix.tolist()) * qr[0]
        result_src = np.array(circuit.get_statevector()).reshape(-1, 1)
        Matrix(R(pi / 2, pi / 2).matrix.tolist()) * qr[0]
        Matrix(R(pi / 2, pi / 2).matrix.tolist()).inv() * qr[0]
        result_des = np.array(circuit.get_statevector()).reshape(-1, 1)
        assert np.allclose(result_src, result_des)


if __name__ == '__main__':
    pytest.main(['-s', 'qutrunk\\test\\gate\\test_r_gate.py'])