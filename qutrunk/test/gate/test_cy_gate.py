import pytest
import numpy as np

from qiskit import QuantumCircuit, BasicAer, transpile

from qutrunk.circuit.gates import CY, All, H
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Matrix
from qutrunk.test.gate.backend_fixture import backend, backend_type

class Test_CY_Gate:
    @pytest.fixture
    def result_gate(self, backend):
        circuit = QCircuit(backend=backend)
        qr = circuit.allocate(2)
        All(H) * qr
        CY * (qr[0], qr[1])
        result_gate = np.array(circuit.get_statevector()).reshape(-1, 1)
        return result_gate

    def test_result_matrix(self, result_gate):
        circuit = QCircuit()
        qr = circuit.allocate(2)
        Matrix(H.matrix.tolist()) * qr[0]
        Matrix(H.matrix.tolist()) * qr[1]
        Matrix(CY.matrix.tolist()) * (qr[0], qr[1])
        result_matrix = np.array(circuit.get_statevector()).reshape(-1, 1)
        assert np.allclose(result_gate, result_matrix)

    def test_result_qiskit(self, result_gate):
        qc = QuantumCircuit(2, 2)
        backend = BasicAer.get_backend('statevector_simulator')
        qc.h(0)
        qc.h(1)
        qc.cy(0, 1)
        job = backend.run(transpile(qc, backend))
        result_qiskit = np.array(job.result().get_statevector(qc)).reshape(-1, 1)
        assert np.allclose(result_gate, result_qiskit)

    def test_gate_inverse(self, backend):
        circuit = QCircuit(backend=backend)
        qr = circuit.allocate(2)
        All(H) * qr
        result_src = np.array(circuit.get_statevector()).reshape(-1, 1)
        CY * (qr[0], qr[1])
        CY.inv() * (qr[0], qr[1])
        result_des = np.array(circuit.get_statevector()).reshape(-1, 1)
        assert np.allclose(result_src, result_des)

    def test_matrix_inverse(self, backend):
        circuit = QCircuit(backend=backend)
        qr = circuit.allocate(2)
        Matrix(H.matrix.tolist()) * qr[0]
        Matrix(H.matrix.tolist()) * qr[1]
        result_src = np.array(circuit.get_statevector()).reshape(-1, 1)
        Matrix(CY.matrix.tolist()) * (qr[0], qr[1])
        Matrix(CY.matrix.tolist()).inv() * (qr[0], qr[1])
        result_des = np.array(circuit.get_statevector()).reshape(-1, 1)
        assert np.allclose(result_src, result_des)


if __name__ == '__main__':
    pytest.main(['-s', 'qutrunk\\test\\gate\\test_cy_gate.py'])