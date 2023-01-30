import pytest
import numpy as np

from qutrunk.circuit.gates import R, All, H
from qutrunk.circuit import QCircuit
from qutrunk.backends import BackendLocal

def test_r_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    All(H) * qr
    R(np.pi / 2, np.pi / 2) * qr[0]
    result = np.array(circuit.get_statevector()).reshape(-1, 1)

    circuit_gpu = QCircuit(backend=BackendLocal("gpu"))
    qr_gpu = circuit_gpu.allocate(2)
    All(H) * qr_gpu
    R(np.pi / 2, np.pi / 2) * qr_gpu[0]
    result_gpu = np.array(circuit_gpu.get_statevector()).reshape(-1, 1)

    assert np.allclose(result, result_gpu)