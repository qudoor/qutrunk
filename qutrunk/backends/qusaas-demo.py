import requests
from thriftpy2 import http, rpc
from thriftpy2.http import THttpHeaderFactory
from thriftpy2.protocol import TBinaryProtocolFactory, TMultiplexedProtocolFactory

from qutrunk.backends import BackendQuSaas, BackendQuSprout
from qutrunk.backends.qusprout.rpcclient import QuSproutApiServer, qusprout, qusproutdata
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import All, H, Measure


def origin_rpc():
    circuit = QCircuit(BackendQuSprout())
    qreg = circuit.allocate(2)
    All(H) * qreg
    All(Measure) * qreg
    res = circuit.run(200)
    print(res.excute_info())
    print(res.get_counts())


def rpc_single():
    proto_fac = TBinaryProtocolFactory()
    proto_fac = TMultiplexedProtocolFactory(proto_fac, "QuSproutServer")
    client = rpc.make_client(qusprout.QuSproutServer, "192.168.170.195", 9091, proto_factory=proto_fac)
    state = client.getProbAmp(qusproutdata.GetProbAmpReq('0c6394c50cf841e59bc7585c943b3b6f', 1))
    print()


def local_by_http():
    # http_client = http.make_client(qusprout.QuSproutServer, port=8081, scheme="https",
    #                                http_header_factory={"token": "xxxxx"})
    http_client = http.make_client(qusprout.QuSproutServer, port=8081)

    api_server = QuSproutApiServer()
    api_server._client = http_client

    backend = BackendQuSprout()
    backend._api_server = api_server

    circuit = QCircuit(backend)
    qreg = circuit.allocate(2)
    All(H) * qreg
    All(Measure) * qreg
    res = circuit.run(200)

    print(res.excute_info())
    print(res.get_counts())


def local_rpc_proxy():
    backend = BackendQuSprout()
    rpc_client = backend._api_server._client
    http_server = http.make_server(qusprout.QuSproutServer, rpc_client, host="localhost", port=8081)
    http_server.serve()


def qusaas_http():
    host = "192.168.166.163"
    port = 9999
    ak = "DZeM8sd4S1NrN863GC9ABOtFCa5syegs2M0A9HmB"
    sk = "HivEzszT1jPFvKy0i4NUMyQCdhS2LC8NPKJo3myjxYkMtNjbR550n6BMd4rcyxBeuFSLxdSW5ZlHNOnLG83BWFBTBTMT1BD1Yr8wTf2d3tISgrtTCspiVcW9JSR84gMN"

    header = THttpHeaderFactory({
        'AUTHORIZATION': f'Bearer {get_access_token(ak, sk)}'
    })
    proto_fac = TBinaryProtocolFactory()
    proto_fac = TMultiplexedProtocolFactory(proto_fac, "QuSproutServer")
    client = http.make_client(qusprout.QuSproutServer,
                              proto_factory=proto_fac,
                              host=host,
                              port=port,
                              scheme="http",
                              path="gateway/qusprout/",
                              http_header_factory=header)

    api_server = QuSproutApiServer()
    api_server._client = client

    backend = BackendQuSprout()
    backend._api_server = api_server

    circuit = QCircuit(backend)
    qreg = circuit.allocate(2)
    All(H) * qreg
    All(Measure) * qreg
    res = circuit.run(200)

    print(res.excute_info())
    print(res.get_counts())


def get_access_token(client_id, client_secret):
    data = {
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': 'client_credentials'
    }
    response = requests.post('http://192.168.166.163:9999/oauth2/token/', data=data).json()
    return response['data']['access_token']


def saas_final():
    ak = "DZeM8sd4S1NrN863GC9ABOtFCa5syegs2M0A9HmB"
    sk = "HivEzszT1jPFvKy0i4NUMyQCdhS2LC8NPKJo3myjxYkMtNjbR550n6BMd4rcyxBeuFSLxdSW5ZlHNOnLG83BWFBTBTMT1BD1Yr8wTf2d3tISgrtTCspiVcW9JSR84gMN"
    circuit = QCircuit(BackendQuSaas(ak, sk))
    qreg = circuit.allocate(2)
    All(H) * qreg
    All(Measure) * qreg
    res = circuit.run(200)
    print(res.excute_info())
    print(res.get_counts())


if __name__ == '__main__':
    origin_rpc()
    # rpc_single()

    # qusaas_http()
    # saas_final()

    # th = Thread(target=rpc_proxy)
    # th.start()

    # by_http()
