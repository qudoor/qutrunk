import numpy as np

from qutrunk.circuit import QCircuit
from qutrunk.circuit.ops import AMP


def test_amplitudes_local():
    """Test set amp for local"""
    qubit_len = 3
    circuit = QCircuit()
    qr = circuit.allocate(qubit_len)

    amp_list = [1 - 2j, 2 + 3j, 3 - 4j, 0.5 + 0.7j]
    start_index = 1
    num_amps = 2
    AMP(amp_list, start_index, num_amps) * qr

    if num_amps > len(amp_list):
        num_amps = len(amp_list)

    result = [0] * (2**qubit_len)
    result[0] = 1

    for i in range(num_amps):
        result[i + 1] = complex(
            qr.circuit.cmds[0].cmdex.amp.reals[i], qr.circuit.cmds[0].cmdex.amp.imags[i]
        )
    # local backend
    result_backend = circuit.get_statevector()

    assert np.allclose(result_backend, result)
