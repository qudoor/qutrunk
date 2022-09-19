import uuid
from functools import partial
from os import path
import locale
from thriftpy2.parser import parser

import thriftpy2
from thriftpy2.protocol import TBinaryProtocolFactory, TMultiplexedProtocolFactory
from thriftpy2.rpc import make_client

from qutrunk.tools.function_time import timefn

PARENT_DIR = path.dirname(__file__)

# open in thriftpy2 doesn't support encoding, replace open with default encoding utf8
# refer to issue https://github.com/Thriftpy/thriftpy2/issues/172
# and pull request https://github.com/Thriftpy/thriftpy2/pull/173
parser.open = partial(open, encoding='utf8')

qusprout = thriftpy2.load(path.join(PARENT_DIR, "idl/qusprout.thrift"))
qusproutdata = thriftpy2.load(path.join(PARENT_DIR, "idl/qusproutdata.thrift"))

# reset parser.open to default open
parser.open = open


class QuSproutApiServer:
    """
    The RPC client connect to QuSprout. Used by BackendQuSprout.

    Args:
        ip: ip address.
        port: port, default: 9090.
    """

    def __init__(self, ip='localhost', port=9090):
        proto_fac = TBinaryProtocolFactory()
        proto_fac = TMultiplexedProtocolFactory(proto_fac, "QuSproutServer")
        self._client = make_client(qusprout.QuSproutServer, ip, port, proto_factory=proto_fac)
        self._taskid = uuid.uuid4().hex

    def close(self):
        self._client.close()

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
        return res.result

    @timefn
    def get_prob_amp(self, index):
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
    def get_prob_outcome(self, qubit, outcome):
        """Get the probability of a specified qubit being measured in the given outcome (0 or 1)

        Args:
            qubit: The specified qubit to be measured.
            outcome: The qubit measure result(0 or 1).

        Returns:
            The probability of target qubit
        """
        req = qusproutdata.GetProbOfOutcomeReq(self._taskid, qubit, outcome)
        res = self._client.getProbOfOutcome(req)
        return res.pro_outcome

    @timefn
    def get_prob_all_outcome(self, qubits):
        """Get outcomeProbs with the probabilities of every outcome of the sub-register contained in qureg.

        Args:
            qubits: The sub-register contained in qureg.

        Returns:
            An array contains probability of target qubits.
        """
        req = qusproutdata.GetProbOfAllOutcomReq(self._taskid, qubits)
        res = self._client.getProbOfAllOutcome(req)
        return res.pro_outcomes

    @timefn
    def get_all_state(self):
        """Get the current state vector of probability amplitudes for a set of qubits.

        Returns:
            Array contains all amplitudes of state vector.
        """
        req = qusproutdata.GetAllStateReq(self._taskid)
        res = self._client.getAllState(req)
        return res.all_state

    @timefn
    def cancel_cmd(self):
        """Cancel current job.
        """
        req = qusproutdata.CancelCmdReq(self._taskid)
        return self._client.cancelCmd(req)

    @timefn
    def qft(self, qubits):
        """Applies the quantum Fourier transform (QFT) to a specific subset of qubits of the register qureg.

        Args:
            qubits: A list of the qubits to operate the QFT upon.
        """
        if qubits:
            req = qusproutdata.ApplyQFTReq(self._taskid, qubits)
            self._client.applyQFT(req)
        else:
            req = qusproutdata.ApplyFullQFTReq(self._taskid)
            self._client.applyFullQFT(req)

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
        req = qusproutdata.GetExpecPauliSumReq(self._taskid, oper_type_list, term_coeff_list)
        res = self._client.getExpecPauliSum(req)
        return res.expect
