from enum import IntEnum
import pytest

from qutrunk.backends import BackendQuSprout, BackendLocal

class BackendType(IntEnum):
    LOCAL = 1
    QU_SPROUT = 2

@pytest.fixture(params=[int(bt) for bt in list(BackendType)])
def backend_type(request):
    return request.param


@pytest.fixture
def backend(backend_type):
    return create_backend(backend_type)


def create_backend(backend_type):
    if backend_type == BackendType.LOCAL:
        be = BackendLocal()
    elif backend_type == BackendType.QU_SPROUT:
        be = BackendQuSprout(ip="192.168.170.195", port=9091)
    else:
        raise ValueError(f"unsupported backend type:{backend_type}")

    return be