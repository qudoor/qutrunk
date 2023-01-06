import os
import sys

from qutrunk.backends import BackendLocal, BackendQuSprout


def backend_from_env():
    be_type = os.getenv("BACKEND_TYPE", "local")
    be_type = be_type.lower()
    if be_type == "local":
        return BackendLocal()
    elif be_type == "qusprout":
        run_mode = os.getenv("RUN_MODE", "cpu")
        backend_ip = os.getenv("BACKEND_IP")
        backend_port = os.getenv("BACKEND_PORT")
        return BackendQuSprout(run_mode, backend_ip, backend_port)
    elif be_type == "aws_braket":
        from qutrunk.backends.braket import BackendAWSDevice
        device_name = os.getenv("DEVICE_NAME", "SV1")
        return BackendAWSDevice(device_name)
        # return BackendAWSLocal()
    else:
        err_msg = "backend type [" + be_type + "] not supported"
        sys.exit(err_msg)
