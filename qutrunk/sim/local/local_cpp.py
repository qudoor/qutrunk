from qutrunk.tools.function_time import timefn

try:
    from qutrunk.sim.local import simulator
except ImportError:  # pragma: no cover
    # windows
    from qutrunk.sim.local.Release import simulator


class BackendLocalCpp:
    """Simulator is a compiler engine which simulates a quantum computer using C++-based kernels."""

    @timefn
    def init(self, qubits, show):
        return simulator.init(qubits, show)

    @timefn
    def set_amplitudes(self, reals, imags):
        return simulator.set_amplitudes(reals, imags)
        
    @timefn
    def send_circuit(self, circuit, final):
        """
        Send the quantum circuit to local backend

        Args:
            circuit: quantum circuit to send
            final: True if quantum circuit finish, default False, \
                when final==True The backend program will release the computing resources
        """
        start = circuit.cmd_cursor
        cmds = circuit.cmds[start:]
        temp_cmds = []
        for cmd in cmds:
            temp_cmd = simulator.Cmd()
            temp_cmd.gate = str(cmd.gate)
            temp_cmd.targets = cmd.targets
            temp_cmd.controls = cmd.controls
            temp_cmd.rotation = cmd.rotation
            temp_cmd.desc = cmd.qasm()
            temp_cmd.inverse = cmd.inverse
            temp_cmds.append(temp_cmd)

        simulator.send_circuit(temp_cmds, final)

    @timefn
    def run(self, shots):
        return simulator.run(shots)

    @timefn
    def get_prob_amp(self, index):
        """
        Get the probability of a state-vector at an index in the full state vector.

        Args:
            index: index in state vector of probability amplitudes

        Returns:
            the probability of target index
        """
        return simulator.getProbOfAmp(index)

    @timefn
    def get_prob_outcome(self, qubit, outcome):
        """
        Get the probability of a specified qubit being measured in the given outcome (0 or 1)

        Args:
            qubit: the specified qubit to be measured
            outcome: the qubit measure result(0 or 1)

        Returns:
            the probability of target qubit
        """
        return simulator.getProbOfOutcome(qubit, outcome)

    @timefn
    def get_prob_all_outcome(self, qubit):
        """
        Get outcomeProbs with the probabilities of every outcome of the sub-register contained in qureg

        Args:
            qubits: the sub-register contained in qureg

        Returns:
            An array contains probability of target qubits
        """
        return simulator.getProbOfAllOutcome(qubit)

    @timefn
    def get_all_state(self):
        """
        Get the current state vector of probability amplitudes for a set of qubits
        """
        return simulator.getAllState()

    @timefn
    def qft(self, qubits):
        """
        Applies the quantum Fourier transform (QFT) to a specific subset of qubits of the register qureg

        Args:
            qubits: a list of the qubits to operate the QFT upon
        """
        if qubits:
            return simulator.apply_QFT(qubits)
        else:
            return simulator.apply_Full_QFT()

    @timefn
    def get_expec_pauli_prod(self, pauli_prod_list):
        """
        Computes the expected value of a product of Pauli operators.

        Args:
            pauli_prod_list: a list contains the indices of the target qubits,\
                the Pauli codes (0=PAULI_I, 1=PAULI_X, 2=PAULI_Y, 3=PAULI_Z) to apply to the corresponding qubits.

        Returns:
            the expected value of a product of Pauli operators.
        """
        temp_pauli_prod_list = []
        for item in pauli_prod_list:
            puali_prod_info = simulator.PauliProdInfo()
            puali_prod_info.oper_type = item["oper_type"]
            puali_prod_info.target = item["target"]
            temp_pauli_prod_list.append(puali_prod_info)

        return simulator.getExpecPauliProd(temp_pauli_prod_list)

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
        return simulator.getExpecPauliSum(oper_type_list, term_coeff_list)
