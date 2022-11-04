from enum import Enum


class BasicGate:
    """Base class of all gates. (Don't use it directly but derive from it)."""

    def __init__(self):
        self.is_inverse = False
        self.name = ""

    # TODO: need to improve.
    def __str__(self):
        """The string description of a quantum logic gate."""
        return ""

    # TODO: need to improve.
    def __or__(self, qubit):
        """Quantum logic gate operation. Overwrite | operator to achieve quantum logic gate operation."""
        pass

    # TODO: need to improve.
    def __mul__(self, qubits):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        pass

    def angles(self):
        """Get angle Args."""
        angle_params = ("rotation", "theta", "phi", "lam", "gamma")
        return [getattr(self, param) for param in angle_params if hasattr(self, param)]

    def commit(self, circuit, cmd):
        """Commit command to circuit.

        Args:
            circuit: The circuit of cmd to committed.
            cmd: The target command committed to circuit.
            statement: The origin statement that generate target cmd.
        """
        circuit.append_cmd(cmd)

    # TODO: need to improve.
    def inv(self):
        """Apply inverse gate."""
        # note: 状态相关方法需要重新生成一个新对象
        raise NotImplementedError

    # TODO: need to improve.
    def ctrl(self, ctrl_cnt=1):
        """Apply controlled gate.
        
        Args:
            ctrl_cnt: The number of control qubits, default: 1.
        """
        # note: 状态相关方法需要重新生成一个新对象
        raise NotImplementedError


class BasicRotateGate(BasicGate):
    """Base class for rotation gate."""

    def __init__(self):
        super().__init__()


class BasicPhaseGate(BasicGate):
    """Base class for phase gate."""

    def __init__(self):
        super().__init__()


# TODO: need to improve.
class Observable:
    """Base class representing observables. It's usually PauliX, PauliY, PauliZ, PauliI."""

    def __call__(self, target):
        """
        Get Observable data.

        Args:
            target: The observed qubit.

        Returns:
            The observed data list, each item contains op type and target qubit, \
                e.g: [{"oper_type": 1, "target": 0}]
        """
        return []


class PauliType(Enum):
    """PauliType."""

    PAULI_I = 0
    PAULI_X = 1
    PAULI_Y = 2
    PAULI_Z = 3

class PauliTerm:
    """Pauli Term consist of Pauli operator and coefficients."""
    def __init__(self, coeff, paulis):
        self.coeff = coeff
        self.paulis = [op.value for op in paulis]

    def padding(self, len, op=PauliType.PAULI_I):
        """Fill paulis by specific pauli operator in len times.
        
        Args:
            len: Times to fill op.
            op: The target operator filled to paulis, defaul: PauliI.
        """
        for i in range(len):
            self.paulis.append(op.value)

class PauliTerms:
    """A list of PauliTerm."""
    def __init__(self):
        self.pauli_term_list = []

    def append(self, term):
        """Append Pauli Term."""
        self.pauli_term_list.append(term)

    def __getitem__(self, index):
        return self.pauli_term_list[index]
        
