import numpy as np
import trace
import sys

from qutrunk.circuit.gates import Toffoli, All, H
from qutrunk.circuit import QCircuit
from qutrunk.backends import BackendLocal

def test_toffoli_gate():
    circuit = QCircuit()
    qr = circuit.allocate(3)
    All(H) * qr
    Toffoli * (qr[0], qr[1], qr[2])
    result = np.array(circuit.get_statevector()).reshape(-1, 1)
    
    circuit_mpi = QCircuit(backend=BackendLocal('mpi'))
    qr_mpi = circuit_mpi.allocate(3)
    All(H) * qr_mpi
    Toffoli * (qr_mpi[0], qr_mpi[1], qr_mpi[2])
    result_mpi = np.array(circuit_mpi.get_statevector()).reshape(-1, 1)

    if circuit_mpi.backend._local_impl.sim.reg.chunk_id == 0:
        if np.allclose(result, result_mpi):
            print('test_iswap_gate: T')
        else:
            print('test_iswap_gate: F')


# define Trace object: trace line numbers at runtime, exclude some modules
tracer = trace.Trace(
    ignoredirs=[sys.prefix, sys.exec_prefix],
    ignoremods=[
        'inspect', 'contextlib', '_bootstrap',
        '_weakrefset', 'abc', 'posixpath', 'genericpath', 'textwrap'
    ],
    trace=0,
    count=1)

# by default trace goes to stdout
# redirect to a different file for each processes
# sys.stdout = open('trace_{:04d}.txt'.format(COMM_WORLD.rank), 'w')

tracer.runfunc(test_toffoli_gate)