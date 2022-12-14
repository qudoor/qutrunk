from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Swap


def test_swap():
    cir = QCircuit()
    qr = cir.allocate(2)
    H * qr[0]

    # print(cir.get_statevector())

    Swap * (qr[0], qr[1])

    print(cir.get_statevector())


import sys
import trace

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

tracer.runfunc(test_swap)
