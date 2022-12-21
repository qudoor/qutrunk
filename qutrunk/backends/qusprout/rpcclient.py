import os
import uuid

from thrift.protocol import TBinaryProtocol, TMultiplexedProtocol
from thrift.transport import TSocket, TTransport

from qutrunk.thrift.qusprout import QuSproutServer
from qutrunk.thrift.qusproutdata import ttypes as qusproutdata
from qutrunk.tools.function_time import timefn


class QuSproutApiServer:
    """
    The RPC client connect to QuSprout. Used by BackendQuSprout.

    Args:
        ip: ip address.
        port: port, default: 9090.
    """

    def __init__(self, ip="localhost", port=9090):
        self._ip = ip
        self._port = port
        self._client, self._transport = self.open("QuSproutServer")
        self._taskid = uuid.uuid4().hex

    def open(self, name):
        socket = TSocket.TSocket(self._ip, self._port)
        transport = TTransport.TBufferedTransport(socket)
        protocol = TBinaryProtocol.TBinaryProtocol(transport)
        protocol = TMultiplexedProtocol.TMultiplexedProtocol(protocol, name)
        client = QuSproutServer.Client(protocol)
        try:
            transport.open()
        except Exception:
            print("QuSprout is not available!")
            os._exit(1)
        return client, transport

    def close(self):
        self._transport.close()

    @timefn
    def init(self, qubits, density, exectype=0):
        """
        Initialize the quantum computing program at backend.

        Args:
            qubits: The number of qubits used in the entire circuit.
            state: Init state.
            value: Init Value.
            density: Enable noise model.
            exectype: SingleProcess: use single calculation node; \
                Mpi: parallel calculation using multiple nodes.
        """
        req = qusproutdata.InitQubitsReq(self._taskid, qubits, density, exectype)
        return self._client.initQubits(req)

    @timefn
    def send_circuit(self, circuit, final):
        """
        Args:
            circuit: Quantum circuit to send.
            final: True if quantum circuit finish, default False, \
            when final==True The backend program will release the computing resources.
        """
        req = qusproutdata.SendCircuitCmdReq(self._taskid, circuit, final)
        res = self._client.sendCircuitCmd(req)
        return res

    @timefn
    def run(self, shots):
        """Run quantum circuit.

        Args:
            shots: Circuit run times, for sampling, default: 1.

        Returns:
            result: The Result object contain circuit running outcome.
        """
        req = qusproutdata.RunCircuitReq(self._taskid, shots)
        res = self._client.run(req)
        return res

    @timefn
    def get_prob(self, index):
        """Get the probability of a state-vector at an index in the full state vector.

        Args:
            index: Index in state vector of probability amplitudes

        Returns:
            The probability of target index
        """
        req = qusproutdata.GetProbAmpReq(self._taskid, index)
        res = self._client.getProbAmp(req)
        return res.amp

    @timefn
    def get_probs(self, qubits):
        """Get all probabilities of circuit.

        Returns:
            An array contains all probabilities of circuit.
        """
        req = qusproutdata.GetProbOfAllOutcomReq(self._taskid, qubits)
        res = self._client.getProbOfAllOutcome(req)
        return res.pro_outcomes

    @timefn
    def get_statevector(self):
        """Get the current state vector of probability amplitudes for a set of qubits.

        Returns:
            Array contains all amplitudes of state vector.
        """
        req = qusproutdata.GetAllStateReq(self._taskid)
        res = self._client.getAllState(req)
        return res.all_state

    @timefn
    def cancel_cmd(self):
        """Cancel current job."""
        req = qusproutdata.CancelCmdReq(self._taskid)
        return self._client.cancelCmd(req)

    @timefn
    def get_expec_pauli_prod(self, pauli_prod_list):
        """Computes the expected value of a product of Pauli operators.

        Args:
            pauli_prod_list: A list contains the indices of the target qubits,\
                the Pauli codes (0=PAULI_I, 1=PAULI_X, 2=PAULI_Y, 3=PAULI_Z) to apply to the corresponding qubits.

        Returns:
            The expected value of a product of Pauli operators.
        """
        req = qusproutdata.GetExpecPauliProdReq(self._taskid, pauli_prod_list)
        res = self._client.getExpecPauliProd(req)
        return res.expect

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
        req = qusproutdata.GetExpecPauliSumReq(
            self._taskid, oper_type_list, term_coeff_list
        )
        res = self._client.getExpecPauliSum(req)
        return res.expect

    def get_rand(self, length, cnt=1):
        """
        generate random number by QuDoor RandomCard integrated in QuSprout

        Args:
            length: length of the random number
            cnt: amount of random number

        Examples:
            .. code-block:: python

                from qutrunk.backends import BackendQuSprout

                be = BackendQuSprout(ip='', port=9091)
                rands = be.get_rand(21, 2)
                print(rands)

        Returns:
            list of random numbers

        """
        cl, trs = self.open("QuSproutServerRand")
        req = qusproutdata.GetRandomReq(
            length, cnt
        )
        res = cl.getRandom(req)
        trs.close()
        return res

