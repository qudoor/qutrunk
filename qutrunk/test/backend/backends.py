from enum import IntEnum

import pytest

from qutrunk.backends import BackendLocal, BackendQuSprout, BackendIBM
from qutrunk.backends.braket import BackendAWSLocal, BackendAWSDevice
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, CNOT, All, Measure


class BackendType(IntEnum):
    LOCAL = 1
    QU_SPROUT = 2
    QU_SAAS = 3
    IBM = 4
    AWS_LOCAL = 5
    AWS_DEVICE = 6


# all backend type
@pytest.fixture(params=[int(bt) for bt in list(BackendType)])
def backend_type(request):
    return request.param


@pytest.fixture
def circuit(backend_type):
    cir = QCircuit(backend=create_backend(backend_type))
    return cir


def create_backend(backend_type):
    if backend_type == BackendType.LOCAL:
        be = BackendLocal()
    elif backend_type == BackendType.QU_SPROUT:
        be = BackendQuSprout()
    elif backend_type == BackendType.IBM:
        be = BackendIBM()
    elif backend_type == BackendType.AWS_LOCAL:
        be = BackendAWSLocal()
    elif backend_type == BackendType.AWS_DEVICE:
        be = BackendAWSDevice()
    else:
        raise ValueError(f"unsupported backend type:{backend_type}")

    return be


# override default all backend_type
@pytest.mark.parametrize("backend_type", [BackendType.LOCAL, BackendType.AWS_LOCAL])
def test_backend(circuit):
    qr = circuit.allocate(2)
    H * qr[0]
    CNOT * (qr[0], qr[1])
    All(Measure) * qr
    print()
    circuit.draw()

    res = circuit.run(shots=100)
    print(circuit.backend.name)
    print(res.get_counts())
