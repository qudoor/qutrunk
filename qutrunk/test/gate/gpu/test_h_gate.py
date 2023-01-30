import pytest
import numpy as np

from qutrunk.circuit.gates import H
from qutrunk.circuit import QCircuit
from qutrunk.backends import BackendLocal

def test_h_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    H * qr[0]
    result = np.array(circuit.get_statevector()).reshape(-1, 1)

    circuit_gpu = QCircuit(backend=BackendLocal("gpu"))
    qr_gpu = circuit_gpu.allocate(2)
    H * qr_gpu[0]
    result_gpu = np.array(circuit_gpu.get_statevector()).reshape(-1, 1)

    assert np.allclose(result, result_gpu)