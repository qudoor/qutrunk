from qutrunk.backends import Backend
from .httpclient import QuSaasApiServer


class BackendQuSaas(Backend):
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

    def __init__(self, ak, sk, run_mode: str = "cpu"):
        """

        Args:
            ak: Access key
            sk: Secret Key
            run_mode: cpu: calculation use single cpu; \
                cpu_mpi: parallel calculation using multiple cpu; \ 
                gpu: calculation use single gpu.

        """
        self.circuit = None
        self.run_mode = run_mode
        self._api_server = QuSaasApiServer(ak, sk)
        self.task_id = self._api_server._taskid

    def send_circuit(self, circuit, final=False):
        """Send the quantum circuit to qusprout backend.

        Args:
            circuit: Quantum circuit to send.
            final: True if quantum circuit finish, default False, \
            when final==True The backend program will release the computing resources.
        """
        start = circuit.cmd_cursor
        stop = len(circuit.cmds)

        cmds = circuit.cmds[start:stop]

        circuit.forward(stop - start)

        exectype = 0
        if self.run_mode == "cpu_mpi":
            exectype = 2
        elif self.run_mode == "gpu":
            exectype = 3
        else:
            exectype = 1
            
        if start == 0:
            res, elapsed = self._api_server.init(
                circuit.num_qubits, circuit.density, exectype
            )
            if self.circuit.counter:
                self.circuit.counter.acc_run_time(elapsed)

        if len(cmds) == 0 and (not final):
            return

        res, elapsed = self._api_server.send_circuit(cmds, final)
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)

    def run(self, shots=1):
        """Run quantum circuit.

        Args:
            shots: Circuit run times, for sampling, default: 1.

        Returns:
            result: The Result object contain circuit running outcome.
        """
        res, elapsed = self._api_server.run(shots)
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)
            self.circuit.counter.finish()

        """
        1 必须释放连接，不然其它连接无法连上服务端
        2 不能放在__del__中，因为对象释放不代表析构函数会及时调用
        """
        self._api_server.close()

        return res

    def get_prob(self, index):
        """Get the probability of a state-vector at an index in the full state vector.

        Args:
            index: Index in state vector of probability amplitudes.

        Returns:
            The probability of target index.
        """
        res, elapsed = self._api_server.get_prob(index)
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)
        return res

    def get_probs(self, qubits):
        """Get outcomeProbs with the probabilities of every outcome of the sub-register contained in qureg.

        Args:
            qubits: The sub-register contained in qureg.

        Returns:
            An array contains probability of target qubits.
        """
        res, elapsed = self._api_server.get_probs(qubits)
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)
        return res

    def get_statevector(self):
        """Get the current state vector of probability amplitudes for a set of qubits.

        Returns:
            Array contains all amplitudes of state vector
        """
        res, elapsed = self._api_server.get_statevector()
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)
        return res

    def get_expec_pauli_prod(self, pauli_prod_list):
        """Computes the expected value of a product of Pauli operators.

        Args:
            pauli_prod_list: A list contains the indices of the target qubits,\
            the Pauli codes (0=PAULI_I, 1=PAULI_X, 2=PAULI_Y, 3=PAULI_Z) to apply to the corresponding qubits.

        Returns:
            The expected value of a product of Pauli operators.
        """

        res, elapsed = self._api_server.get_expec_pauli_prod(pauli_prod_list)
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)
        return res

    def get_expec_pauli_sum(self, oper_type_list, term_coeff_list):
        """Computes the expected value of a sum of products of Pauli operators.

        Args:
            oper_type_list: A list of the Pauli codes (0=PAULI_I, 1=PAULI_X, 2=PAULI_Y, 3=PAULI_Z) \
                of all Paulis involved in the products of terms. A Pauli must be specified \
                for each qubit in the register, in every term of the sum.
            term_coeff_list: The coefficients of each term in the sum of Pauli products.

        Returns:
            The expected value of a sum of products of Pauli operators.
        """
        res, elapsed = self._api_server.get_expec_pauli_sum(
            oper_type_list, term_coeff_list
        )
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)
        return res

    def get_rand(self, length, cnt=1):
        """
        generate random number by QuDoor RandomCard integrated in QuSprout

        Args:
            length: length of the random number
            cnt: amount of random number

        Examples:
            .. code-block:: python

                from qutrunk.backends import BackendQuSaas

                ak = "wdHXNTxsFegLUfUfXQZbx44XIkcsnQlxtgL4QTM9"
                sk = "odhprioQ22Iwrump3l3qGWbTfcDMAl1YKO2TAwT5nRLQDWKOFZ6EYVQflRMiw4cJCCEdDLy4FeTPfdy39BeNvo115jRFSiO5YGrfljBUqVHYCq9NWeP6pOGgVyCggQxj"
                be = BackendQuSaas(ak, sk)
                rands = be.get_rand(21, 2)
                print(rands)

        Returns:
            list of random numbers

        """
        return self._api_server.get_rand(length, cnt)

    @property
    def name(self):
        return "BackendQuSaas"
