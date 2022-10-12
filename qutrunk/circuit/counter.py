"""Resources required for the statistics of quantum circuits:
number of quantum bits, number of quantum gates; Quantum circuits are time-consuming to compile and run.
"""
import time


class Counter:
    """Resources of circuit.

    Args:
        circuit: A quantum circuit to statistics the used resource.
    """

    def __init__(self, circuit):
        self.circuit = circuit
        self.qubits = 0
        self.quantum_gates = 0
        self.start_time = time.time()
        self.total_time = 0
        self.backend_time = 0
        self.qutrunk_time = 0

    def acc_run_time(self, elapsed):
        """Accumulate backend running time.

        Args:
            elapsed: The elapsed time.
        """
        self.backend_time += elapsed

    def finish(self):
        """Statistics time and gates when circuit running finish."""
        self.total_time = time.time() - self.start_time
        self.qutrunk_time = self.total_time - self.backend_time
        self.quantum_gates = self.circuit.num_gates

    def __repr__(self):
        return f"Counter(quit={self.qubits})"

    def show_verbose(self):
        """Print the resource info of circuit."""
        print("==================Counter==================")
        print(self)
        print("qubits =", self.qubits)
        print("quantum_gates =", self.quantum_gates)
        print("total_time =", self.total_time)
        print("qutrunk_time =", self.qutrunk_time)
        print("backend_time =", self.backend_time)
