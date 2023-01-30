import pytest
import numpy as np

from qutrunk.circuit.gates import SqrtXdg, All, H
from qutrunk.circuit import QCircuit
from qutrunk.backends import BackendLocal

def test_SqrtXdg_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    All(H) * qr
    SqrtXdg * qr[0]
    result = np.array(circuit.get_statevector()).reshape(-1, 1)

    circuit_gpu = QCircuit(backend=BackendLocal("gpu"))
    qr_gpu = circuit_gpu.allocate(2)
    All(H) * qr_gpu
    SqrtXdg * qr_gpu[0]
    result_gpu = np.array(circuit_gpu.get_statevector()).reshape(-1, 1)

    assert np.allclose(result, result_gpu)
