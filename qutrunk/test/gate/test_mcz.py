import pytest
import numpy as np

from qiskit import QuantumCircuit, BasicAer, transpile

from qutrunk.circuit.gates import MCZ, All, H
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Matrix
from qutrunk.test.gate.backend_fixture import backend, backend_type

class TestCXGate:
    @pytest.fixture
    def result_gate(self, backend):
        circuit = QCircuit(backend=backend)
        qr = circuit.allocate(3)
        All(H) * qr
        MCZ(2) * (qr[0], qr[1], qr[2])
        result_gate = np.array(circuit.get_statevector()).reshape(-1, 1)
        return result_gate

    def test_result_matrix(self, result_gate):
        circuit = QCircuit()
        qr = circuit.allocate(3)
        All(Matrix(H.matrix.tolist())) * qr
        Matrix(MCZ(2).matrix.tolist()) * (qr[0], qr[1], qr[2])
        result_matrix = np.array(circuit.get_statevector()).reshape(-1, 1)
        assert np.allclose(result_gate, result_matrix)

    def test_result_qiskit(self, result_gate):
        qc = QuantumCircuit(3, 3)
        backend = BasicAer.get_backend('statevector_simulator')
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.ccz(0, 1, 2)
        job = backend.run(transpile(qc, backend))
        result_qiskit = np.array(job.result().get_statevector(qc)).reshape(-1, 1)
        assert np.allclose(result_gate, result_qiskit)

    def test_gate_inverse(self, backend):
        circuit = QCircuit(backend=backend)
        qr = circuit.allocate(3)
        All(H) * qr
        result_src = np.array(circuit.get_statevector()).reshape(-1, 1)
        MCZ(2) * (qr[0], qr[1], qr[2])
        MCZ(2).inv() * (qr[0], qr[1], qr[2])
        result_des = np.array(circuit.get_statevector()).reshape(-1, 1)
        assert np.allclose(result_src, result_des)

    def test_matrix_inverse(self, backend):
        circuit = QCircuit(backend=backend)
        qr = circuit.allocate(3)
        All(Matrix(H.matrix.tolist())) * qr
        result_src = np.array(circuit.get_statevector()).reshape(-1, 1)
        Matrix(MCZ(2).matrix.tolist()) * (qr[0], qr[1], qr[2])
        Matrix(MCZ(2).matrix.tolist()).inv() * (qr[0], qr[1], qr[2])
        result_des = np.array(circuit.get_statevector()).reshape(-1, 1)
        assert np.allclose(result_src, result_des)