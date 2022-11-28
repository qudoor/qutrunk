"""Python implementation of a quantum computer simulator."""

from qutrunk.backends.backend import Backend
# TODO:need to improve.
from .local_python import BackendLocalPython as BackendLocalImpl


class BackendLocal(Backend):
    """Python implementation of a quantum computer simulator.

    The local backend uses the simulator to run the quantum circuit.

    Example:
        .. code-block:: python

            from qutrunk.backends import BackendLocal
            from qutrunk.circuit import QCircuit
            from qutrunk.circuit.gates import H, CNOT, Measure

            # new QCircuit object
            qc = QCircuit(backend=BackendLocal())
            # or use as default
            # qc = QCircuit()
            qr = qc.allocate(2)
            H * qr[0]
            CNOT * (qr[0], qr[1])
            Measure * qr[0]
            Measure * qr[1]
            res = qc.run(shots=100)
    """

    def __init__(self):
        super().__init__()
        self.circuit = None
        self._local_impl = BackendLocalImpl()

    def send_circuit(self, circuit, final=False):
        """Send the quantum circuit to local backend.

        Args:
            circuit: Quantum circuit to send.
            final: True if quantum circuit finish, default False, \
            when final==True The backend program will release the computing resources.
        """
        start = circuit.cmd_cursor
        stop = len(circuit.cmds)

        if start == 0:
            res, elapsed = self._local_impl.init(len(circuit.qreg))
            if self.circuit.counter:
                self.circuit.counter.acc_run_time(elapsed)

        res, elapsed = self._local_impl.send_circuit(circuit, final)

        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)

        circuit.forward(stop - start)

        return res

    def run(self, shots=1):
        """Run quantum circuit.

        Args:
            shots: Circuit run times, for sampling, default: 1.

        Returns:
            list: The Result object contain circuit running outcome.
        """
        res, elapsed = self._local_impl.run(shots)
        # TODO: circuit is None?
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)
            self.circuit.counter.finish()
        return res

    def get_prob(self, value):
        """Get probability of the possible measure result of circuit.

        Args:
            value: The target value.

        Returns:
            float:The probability of target index.
        """
        res, elapsed = self._local_impl.get_prob(value)
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)
        return res

    def get_probs(self, qubits):
        """Get all probabilities of circuit.

        Returns:
            list: An array contains all probabilities of circuit.
        """
        res, elapsed = self._local_impl.get_probs(qubits)
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)
        return res

    def get_statevector(self):
        """Get state vector of circuit.

        Returns:
            list: Array contains all amplitudes of state vector.
        """
        res, elapsed = self._local_impl.get_statevector()
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)

        return res

    def get_expec_pauli_prod(self, pauli_prod_list):
        """Computes the expected value of a product of Pauli operators.

        Args:
            pauli_prod_list: A list contains the indices of the target qubits,\
                the Pauli codes (0=PAULI_I, 1=PAULI_X, 2=PAULI_Y, 3=PAULI_Z) to apply to the corresponding qubits.

        Returns:
            float:The expected value of a product of Pauli operators.
        """
        res, elapsed = self._local_impl.get_expec_pauli_prod(pauli_prod_list)
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
            float:The expected value of a sum of products of Pauli operators.
        """
        res, elapsed = self._local_impl.get_expec_pauli_sum(
            oper_type_list, term_coeff_list
        )
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)
        return res

    @property
    def name(self):
        """The name of Backend."""
        return "BackendLocalPython"
