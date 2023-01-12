import pytest
import numpy as np

from qutrunk.circuit.gates import CH, All, H
from qutrunk.circuit import QCircuit
from qutrunk.backends import BackendLocal

def test_ch_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    All(H) * qr
    CH * (qr[0], qr[1])
    result = np.array(circuit.get_statevector()).reshape(-1, 1)

    # circuit_gpu = QCircuit(backend=BackendLocal("gpu"))
    # qr_gpu = circuit_gpu.allocate(2)
    # All(H) * qr_gpu
    # CH * (qr_gpu[0], qr_gpu[1])
    # result_gpu = np.array(circuit_gpu.get_statevector()).reshape(-1, 1)

    # assert np.allclose(result, result_gpu)

test_ch_gate()