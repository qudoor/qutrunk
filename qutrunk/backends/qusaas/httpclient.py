import base64
import json
import uuid

import requests

from qutrunk.config import configuration
from qutrunk.tools.function_time import timefn
from qutrunk.backends.result import MeasureQubit, MeasureQubits, MeasureResult


class QuSaasApiServer:
    """The http client connect to QuSaas. Used by BackendQuSaas.
    """

    def __init__(self, ak, sk):
        """

        Args:
            ak: Access key, refer to quSaas site to get ak/sk, https://xxx
            sk: Secret key, refer to quSaas site to get ak/sk, https://xxx

        """
        pot_conf = configuration['QuPot']
        self.base_url = f"{pot_conf['schema']}://{pot_conf['host']}:{pot_conf['port']}{pot_conf['api']['common']}"
        self._client = requests.Session()
        self._client.hooks["response"].append(
            lambda r, *i_args, **i_kwargs: r.raise_for_status()
        )
        headers = {
            'AUTHORIZATION': f'Bearer {self.get_access_token(ak, sk)}',
            "Content-Type": "application/json; charset=utf8",
        }

        self._client.headers.update(headers)
        self._taskid = uuid.uuid4().hex

    def close(self):
        self._client.close()

    @timefn
    def init(self, qubits, density, exec_type=0):
        """
        Initialize the quantum computing program at backend.

        Args:
            qubits: The number of qubits used in the entire circuit.
            density: Enable noise model.
            exec_type:
                SingleProcess: use single calculation node; \
                Mpi: parallel calculation using multiple nodes.

        """
        req = json.dumps({
            "flowid": "1",
            "taskid": self._taskid,
            "cmd": "initenv",
            "params": {
                "qubits": qubits,
                "density": density,
                "exec_type": exec_type
            },
        })
        res = self._client.post(self.base_url, data=req)
        return res.json()['code']

    @timefn
    def send_circuit(self, cmds, final):
        """
        Args:
            cmds: Quantum circuit to send.
            final: True if quantum circuit finish, default False, \
                when final==True The backend program will release the computing resources.

        """
        circuits = []
        for cmd in cmds:
            a_cmd = {'gate': str(cmd.gate),
                     'targets': cmd.targets,
                     'controls': cmd.controls,
                     'rotations': cmd.rotation,
                     'qasmdef': cmd.qasm(),
                     'inverse': int(cmd.inverse)}
            circuits.append(a_cmd)
        req = json.dumps({
            "flowid": "2",
            "taskid": self._taskid,
            "cmd": "addcmd",
            "params": {
                "final": int(final),
                "circuits": circuits,
            },
        })

        res = self._client.post(self.base_url, req)
        return res.json()["code"]

    @timefn
    def run(self, shots):
        """Run quantum circuit.

        Args:
            shots: Circuit run times, for sampling, default: 1.

        Returns:
            result: The Result object contain circuit running measure.
        """
        req = json.dumps({
            "flowid": "3",
            "taskid": self._taskid,
            "cmd": "runcmd",
            "params": {
                "shots": shots,
            },
        })
        resp = self._client.post(self.base_url, req)
        resp = resp.json()

        result = MeasureResult()
        if resp["data"] is not None and resp["data"]["measures"] is not None:
            for meas in resp["data"]["measures"]:
                meas_temp = MeasureQubits()
                for mea in meas["measure"]:
                    mea_temp = MeasureQubit(mea["target"], mea["value"])
                    meas_temp.measure.append(mea_temp)
                result.measures.append(meas_temp)
        return result

    @timefn
    def get_prob(self, index):
        """Get the probability of a state-vector at an index in the full state vector.

        Args:
            index: Index in state vector of probability amplitudes

        Returns:
            The probability of target index
        """
        req = json.dumps({
            "flowid": "4",
            "taskid": self._taskid,
            "cmd": "getamp",
            "params": {
                "indexs": [index],
            },
        })
        res = self._client.post(self.base_url, req)
        return res.json()["data"]["amps"][0]

    @timefn
    def get_probs(self, qubits):
        """Get outcomeProbs with the probabilities of every outcome of the sub-register contained in qureg.

        Args:
            qubits: The sub-register contained in qureg.

        Returns:
            An array contains probability of target qubits.
        """
        req = json.dumps({
            "flowid": "5",
            "taskid": self._taskid,
            "cmd": "getprob",
            "params": {
                "targets": qubits,
            },
        })
        res = self._client.post(self.base_url, req)
        return res.json()["data"]["outcomes"]

    @timefn
    def get_statevector(self):
        """Get the current state vector of probability amplitudes for a set of qubits.

        Returns:
            Array contains all amplitudes of state vector.
        """
        req = json.dumps({
            "flowid": "6",
            "taskid": self._taskid,
            "cmd": "getstate",
        })
        res = self._client.post(self.base_url, req)
        return res.json()['data']['states']

    @timefn
    def cancel_cmd(self):
        """Cancel current job."""
        req = json.dumps({
            "flowid": "7",
            "taskid": self._taskid,
            "cmd": "releaseenv",
        })
        res = self._client.post(self.base_url, req)
        return res.json()['code']

    @timefn
    def get_expec_pauli_prod(self, pauli_prod_list):
        """Computes the expected value of a product of Pauli operators.

        Args:
            pauli_prod_list: A list contains the indices of the target qubits,\
                the Pauli codes (0=PAULI_I, 1=PAULI_X, 2=PAULI_Y, 3=PAULI_Z) to apply to the corresponding qubits.

        Returns:
            The expected value of a product of Pauli operators.
        """
        req = json.dumps({
            "flowid": "9",
            "taskid": self._taskid,
            "cmd": "getpauli",
            "params": {
                "paulis": pauli_prod_list
            }
        })
        res = self._client.post(self.base_url, req)
        return res.json()['data']['expect']

    @timefn
    def get_expec_pauli_sum(self, oper_type_list, term_coeff_list):
        """
        Computes the expected value of a sum of products of Pauli operators.

        Args:
            oper_type_list: a list of the Pauli codes (0=PAULI_I, 1=PAULI_X, 2=PAULI_Y, 3=PAULI_Z) \
                of all Paulis involved in the products of terms. A Pauli must be specified \
                for each qubit in the register, in every term of the sum.
            term_coeff_list: the coefficients of each term in the sum of Pauli products.

        Returns:
            the expected value of a sum of products of Pauli operators.
        """
        req = json.dumps({
            "flowid": "10",
            "taskid": self._taskid,
            "cmd": "getpaulisum",
        })
        res = self._client.post(self.base_url, req)
        return res.json()['data']['expect']

    def get_rand(self, length, cnt):
        """
        generate random number by QuDoor RandomCard integrated in QuSprout

        Args:
            length: length of the random number
            cnt: amount of random number

        Returns:

        """
        req = json.dumps({
            "flowid": "11",
            "cmd": "getrand",
            "params": {
                "randomlength": length,
                "randomnum": cnt
            }
        })
        res = self._client.post(self.base_url, req)
        return [base64.b64decode(b64str) for b64str in res.json()['data']['randoms']]

    def get_access_token(self, ak, sk):
        """

        Returns:
            dynamic token for qusaas

        """
        data = {
            'client_id': ak,
            'client_secret': sk,
            'grant_type': 'client_credentials'
        }
        saas_conf = configuration['QuSaas']
        url = f"{saas_conf['schema']}://{saas_conf['host']}:{saas_conf['port']}{saas_conf['api']['token']}"
        response = self._client.post(url, data=data).json()
        return response['access_token']
