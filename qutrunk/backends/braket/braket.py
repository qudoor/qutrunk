"""AWS Braket backends."""

import datetime
from abc import ABC

from braket.aws import AwsDevice
from braket.circuits import Circuit
from braket.device_schema.dwave import DwaveDeviceCapabilities
from braket.device_schema.xanadu import XanaduDeviceCapabilities
from braket.devices import LocalSimulator
from braket.tasks.local_quantum_task import LocalQuantumTask

from .adapter import convert_qutrunk_to_braket_circuit
from .braket_job import AWSBraketJob
from .exception import QuTrunkBraketException
from .. import Backend

CLIENT_VERSION = "0.0.1"


class BackendBraket(Backend, ABC):
    """BraketBackend."""

    def __init__(
            self,
            device,
            name: str = None,
            description: str = None,
            online_date: datetime.datetime = None,
    ):
        """Initialize a based backend

        Args:
            device: device: Braket device class
            online_date:  online date
            name: An optional name for the backend
            description: An optional description of the backend

        Raises:
            AttributeError: If a field is specified that's outside the backend's
                options
        """
        self._device = device
        self._name = name
        self._description = description
        self._online_date = online_date
        self._status = device.status
        self._circuit = None

    def send_circuit(self, circuit, final=False):
        self._circuit = circuit

    def run(self, shots=1024):
        circuit: Circuit = convert_qutrunk_to_braket_circuit(self._circuit)
        try:
            task: LocalQuantumTask = self._device.run(
                task_specification=circuit, shots=shots
            )
        except Exception as ex:
            task.cancel()
            raise ex

        return AWSBraketJob(
            job_id=task.id,
            backend=self,
            task=task,
            shots=shots,
        ).result()

    @property
    def name(self):
        return f"BackendAWSBraket[{self._name}]"


class BackendAWSLocal(BackendBraket):
    """BraketLocalBackend."""

    def __init__(self, name: str = "default"):
        """AWSBraketLocalBackend for local execution of circuits.

        Example:
            .. code-block:: python

                device = LocalSimulator()                         #Local State Vector Simulator
                device = LocalSimulator("default")                #Local State Vector Simulator
                device = LocalSimulator(name="default")        #Local State Vector Simulator
                device = LocalSimulator(name="braket_sv")      #Local State Vector Simulator
                device = LocalSimulator(name="braket_dm")      #Local Density Matrix Simulator

        Args:
            name: name of backend
        """
        device = LocalSimulator(backend=name)
        super().__init__(device, name="sv_simulator")


class BackendAWSDevice(BackendBraket):
    """ AWSBraketBackend.
        following below steps to use AWS braket device.
        referring to https://docs.aws.amazon.com/braket/index.html for detail
        1. sign in AWS console, visit https://console.aws.amazon.com/braket/home.
            follow guide to create execution roles, including service-link role and job execute role
        2. set up AWS credentials and default region,
            referring to https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html
    """

    def __init__(self, name: str = None):
        """AWSBraketBackend for execution circuits against AWS Braket devices.

        Args:
            name: name of backend, such as SV1, TN1, DM1, etc. refer to Amazon Braket service for detail
        """
        device = supported_aws_device(name)[0]
        user_agent = f"QuTrunkBraketProvider/{CLIENT_VERSION}"
        device.aws_session.add_braket_user_agent(user_agent)
        super().__init__(
            device,
            name=name,
            description=f"AWS Device: {device.provider_name} {device.name}.",
            online_date=device.properties.service.updatedAt,
        )


def supported_aws_device(name):
    """

    Args:
        name: device name, such as
         [Aspen-10],
         [Aspen-11],
         [Aspen-8],
         [Aspen-9],
         [Aspen-M-1],
         [IonQ Device],
         [Lucy],
         [SV1],
         [TN1],
         [dm1]]

    Returns:
        supported device, such as
         [BraketBackend[Aspen-10],
         BraketBackend[Aspen-11],
         BraketBackend[Aspen-8],
         BraketBackend[Aspen-9],
         BraketBackend[Aspen-M-1],
         BraketBackend[IonQ Device],
         BraketBackend[Lucy],
         BraketBackend[SV1],
         BraketBackend[TN1],
         BraketBackend[dm1]]

    """
    names = [name] if name else None
    devices = AwsDevice.get_devices(names=names)
    supported_devices = [
        d
        for d in devices
        if not isinstance(
            d.properties, (DwaveDeviceCapabilities, XanaduDeviceCapabilities)
        )
    ]

    if not supported_devices:
        raise QuTrunkBraketException(f"no available aws device for '{name}'")
    return supported_devices
