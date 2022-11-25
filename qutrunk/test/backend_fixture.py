from enum import IntEnum

import pytest

from qutrunk.backends import BackendLocal, BackendQuSprout, BackendIBM
from qutrunk.backends.braket import BackendAWSLocal, BackendAWSDevice
from qutrunk.circuit import QCircuit


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
        backend = BackendLocal()
    elif backend_type == BackendType.QU_SPROUT:
        backend = BackendQuSprout()
    elif backend_type == BackendType.IBM:
        backend = BackendIBM()
    elif backend_type == BackendType.AWS_LOCAL:
        backend = BackendAWSLocal()
    elif backend_type == BackendType.AWS_DEVICE:
        backend = BackendAWSDevice()
    else:
        raise ValueError(f"unsupported backend type:{backend_type}")

    return backend
