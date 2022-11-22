import numpy as np
from numpy import pi

from qutrunk.circuit.gates import CP, X
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates.meta import Matrix
from qutrunk.backends import BackendQuSprout

from qiskit import QuantumCircuit, BasicAer, transpile

class Test_CP:
    def setup_class(self):
        # qiskit
        qc = QuantumCircuit(2, 2)
        backend = BasicAer.get_backend('statevector_simulator')
        qc.x(0)
        qc.cp(pi/2, 0 ,1)
        job = backend.run(transpile(qc, backend))
        qc_state = job.result().get_statevector(qc)
        self.qiskit_state = np.array(qc_state).reshape(-1, 1)

    def teardown_class(self):
        None

    def test_backend(self):
        # gate
        circuit = QCircuit()
        qr = circuit.allocate(2)
        X * qr[0]
        CP(pi / 2) * (qr[0], qr[1])
        gate_state = np.array(circuit.get_statevector()).reshape(-1, 1)
        assert np.allclose(self.qiskit_state, gate_state)

        # matrix
        _circuit = QCircuit(backend=BackendQuSprout())
        _qr = _circuit.allocate(2)
        Matrix(CP(pi / 2).matrix.tolist()) * (_qr[0], _qr[1])
        matrix_state = np.array(_circuit.get_statevector()).reshape(-1, 1)
        assert np.allclose(self.qiskit_state, matrix_state)

    def test_inverse(self):
        # gate
        circuit = QCircuit()
        qr = circuit.allocate(2)
        pre_gate_state = np.array(circuit.get_statevector()).reshape(-1, 1)
        X * qr[0]
        CP(pi / 2) * (qr[0], qr[1])
        CP(pi / 2).inv() * (qr[0], qr[1])
        X * qr[0]
        post_gate_state = np.array(circuit.get_statevector()).reshape(-1, 1)
        assert np.allclose(pre_gate_state, post_gate_state)

        # matrix
        _circuit = QCircuit()
        _qr = _circuit.allocate(2)
        pre_matrix_state = np.array(_circuit.get_statevector()).reshape(-1, 1)
        Matrix(CP(pi / 2).matrix.tolist()) * (_qr[0], _qr[1])
        Matrix(CP(pi / 2).matrix.tolist()).inv() * (_qr[0], _qr[1])
        post_matrix_state = np.array(_circuit.get_statevector()).reshape(-1, 1)
        assert np.allclose(pre_matrix_state, post_matrix_state)
