import pytest
import numpy as np

from qutrunk.circuit.gates import Reset, All, H, CNOT
from qutrunk.circuit import QCircuit
from qutrunk.backends import BackendLocal

def test_reset_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    All(H) * qr
    CNOT * (qr[0], qr[1])
    Reset * qr
    result = np.array(circuit.get_statevector()).reshape(-1, 1)

    circuit_gpu = QCircuit(backend=BackendLocal("gpu"))
    qr_gpu = circuit_gpu.allocate(2)
    All(H) * qr_gpu
    CNOT * (qr_gpu[0], qr_gpu[1])
    Reset * qr_gpu
    result_gpu = np.array(circuit_gpu.get_statevector()).reshape(-1, 1)

    assert np.allclose(result, result_gpu)