import os
import uuid

import requests
import yaml
from thriftpy2.http import THttpHeaderFactory, make_client
from thriftpy2.protocol import TBinaryProtocolFactory, TMultiplexedProtocolFactory

from qutrunk.tools.get_config import BASE_DIR
from .qusprout import BackendQuSprout, ExecType, QuSproutApiServer, qusprout

API_PREFIX = "gateway/qusprout/"
API_TOKEN = "/oauth2/token/"


class BackendQuSaas(BackendQuSprout):
    """
    QuSaas: quantum circuit simulator wrapped in quSaas,
    provide multi-threaded OMP, multi node parallel MPI, GPU hardware acceleration.
    To use QuSaas, make sure the network is connected and the service host and Port are set correctly.

    Example:
        .. code-block:: python

            from qutrunk.circuit import QCircuit
            from qutrunk.backends import BackendQuSaas
            from qutrunk.circuit.gates import H, CNOT, Measure

            # refer to quSaas site to get ak/sk, https://xxx
            ak = "DZeM8sd4S1NrN863GC9ABOtFCa5syegs2M0A9HmB"
            sk = "HivEzszT1jPFvKy0i4NUMyQCdhS2LC8NPKJo3myjxYkMtNjbR550n6BMd4rcyxBeuFSLxdSW5ZlHNOnLG83BWFBTBTMT1BD1Yr8wTf2d3tISgrtTCspiVcW9JSR84gMN"

            # use BackendQuSaas
            qc = QCircuit(BackendQuSaas(ak, sk))
            qr = qc.allocate(2)

            # apply gate
            H * qr[0]
            CNOT * (qr[0], qr[1])
            Measure * qr[0]
            Measure * qr[1]

            # run circuit
            res = qc.run(shots=100)

            # print result
            print(res.get_counts())"""

    def __init__(self, ak, sk, exectype=ExecType.SingleProcess, ):
        """

        Args:
            ak: Access key
            sk: Secret Key
            exectype:
                SingleProcess: use single calculation node;
                Mpi: parallel calculation using multiple nodes.

        """
        self.circuit = None
        self.exectype = exectype
        self._api_server = QuSaasApiServer(get_saas_config(), ak, sk)


class QuSaasApiServer(QuSproutApiServer):
    """The http client connect to QuSaas. Used by BackendQuSaas.
    """

    def __init__(self, saas_config, ak, sk):
        """

        Args:
            saas_config: 
                schema: QuSaas service scheme
                host: QuSaas service host
                port:QuSaas service port
            ak: Access key, refer to quSaas site to get ak/sk, https://xxx
            sk: Secret key, refer to quSaas site to get ak/sk, https://xxx

        """
        self.scheme = saas_config["scheme"]
        self.host = saas_config["host"]
        self.port = saas_config["port"]
        self.ak = ak
        self.sk = sk

        header = THttpHeaderFactory({
            'AUTHORIZATION': f'Bearer {self.get_access_token()}'
        })
        proto_fac = TBinaryProtocolFactory()
        proto_fac = TMultiplexedProtocolFactory(proto_fac, "QuSproutServer")
        self._client = make_client(qusprout.QuSproutServer,
                                   proto_factory=proto_fac,
                                   host=self.host,
                                   port=self.port,
                                   scheme=self.scheme,
                                   path=API_PREFIX,
                                   http_header_factory=header)
        self._taskid = uuid.uuid4().hex

    def close(self):
        self._client.close()

    def get_access_token(self):
        """

        Returns:
            dynamic token for qusaas

        """
        data = {
            'client_id': self.ak,
            'client_secret': self.sk,
            'grant_type': 'client_credentials'
        }
        url = f"{self.scheme}://{self.host}:{self.port}{API_TOKEN}"
        response = requests.post(url, data=data).json()
        return response['data']['access_token']


def get_saas_config():
    """

    Returns:
        QuSaas config

    """
    path = os.path.join(BASE_DIR, "config/qusaas.yaml")
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config
