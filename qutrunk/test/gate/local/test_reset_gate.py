import pytest
import numpy as np

from qiskit import QuantumCircuit, BasicAer, transpile

from qutrunk.circuit.gates import All, H, CNOT, Reset
from qutrunk.circuit import QCircuit
from qutrunk.test.gate.local.backend_fixture import backend, backend_type

class TestRGate:
    @pytest.fixture
    def result_gate(self, backend):
        circuit = QCircuit(backend=backend)
        qr = circuit.allocate(2)
        All(H) * qr
        CNOT * (qr[0], qr[1])
        Reset * qr
        result_gate = np.array(circuit.get_statevector()).reshape(-1, 1)
        return result_gate

    def test_result_qiskit(self, result_gate):
        qc = QuantumCircuit(2, 2)
        backend = BasicAer.get_backend('statevector_simulator')
        qc.h(0)
        qc.h(1)
        qc.cnot(0, 1)
        qc.reset(0)
        qc.reset(1)
        job = backend.run(transpile(qc, backend))
        result_qiskit = np.array(job.result().get_statevector(qc)).reshape(-1, 1)
        assert np.allclose(result_gate, result_qiskit)