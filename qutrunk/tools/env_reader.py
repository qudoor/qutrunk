import os
import sys

from qutrunk.backends import BackendLocal, BackendQuSprout


def backend_from_env():
    be_type = os.getenv("backend_type", "local")
    be_type = be_type.lower()
    if be_type == "local":
        return BackendLocal()
    elif be_type == "qusprout":
        run_mode = os.getenv("run_mode", "cpu")
        backend_ip = os.getenv("backend_ip")
        backend_port = os.getenv("backend_port")
        return BackendQuSprout(run_mode, backend_ip, backend_port)
    elif be_type == "aws_braket":
        from qutrunk.backends.braket import BackendAWSDevice
        device_name = os.getenv("device_name", "SV1")
        return BackendAWSDevice(device_name)
        # return BackendAWSLocal()
    else:
        err_msg = "backend type [" + be_type + "] not supported"
        sys.exit(err_msg)
