import os
import uuid

from thrift.protocol import TBinaryProtocol, TMultiplexedProtocol
from thrift.transport import TSocket, TTransport

from qutrunk.sim.qusprout.qusprout import QuSproutServer
from qutrunk.sim.qusprout.qusproutdata import ttypes as qusproutdata
from qutrunk.tools.function_time import timefn


class QuSproutApiServer:
    """
    The RPC client connect to QuSprout. Used by BackendQuSprout.

    Args:
        ip: ip address.
        port: port, default: 9090.
    """

    def __init__(self, ip='localhost', port=9090):
        socket = TSocket.TSocket(ip, port)
        self._transport = TTransport.TBufferedTransport(socket)
        protocol = TBinaryProtocol.TBinaryProtocol(self._transport)
        quest = TMultiplexedProtocol.TMultiplexedProtocol(protocol, "QuSproutServer")
        self._client = QuSproutServer.Client(quest)
        self._taskid = uuid.uuid4().hex
        try:
            self._transport.open()
        except Exception:
            print("QuSprout is not available!")
            os._exit(1)

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
