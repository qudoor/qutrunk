"""Command Module."""


class Command:
    """Converts the quantum gate operation into a specific command.

    Args:
        gate: Quantum gate operator.
        targets: Target qubits.
        controls: Control qubits.
        rotation: Angle for rotation gate.
        inverse: Whether to enable the inverse circuit.
    """

    def __init__(self, gate, targets, controls=None, rotation=None, inverse=False):
        # TODO: modify controls and rotation to tuple?
        if controls is None:
            self.controls = []
        else:
            self.controls = controls

        if rotation is None:
            self.rotation = []
        else:
            self.rotation = rotation

        self.gate = gate
        self.targets = targets
        self.cmd_ver = "1.0"
        self.inverse = inverse

    def __eq__(self, other):
        """Two command are the same if they have the same qasm."""
        # TODO: need to improve
        if type(self) is not type(other):
            return False

        if self.qasm() == other.qasm():
            return True
        return False

    def __repr__(self) -> str:
        """Generate a representation of the command object instance.

        Returns:
            str: A representation of the command instance.
        """
        return (
            f"Command(gate={self.gate}, controls={self.controls}, targets={self.targets}, "
            f"rotation={self.rotation}), inverse={self.inverse})"
        )

    def qasm(self) -> str:
        """Generate OpenQASM code for command."""
        name = str(self.gate).lower()

        if str(self.gate) == "Measure":
            index = self.targets[0]
            return f"{name} q[{str(index)}] -> c[{str(index)}]"

        ctrl_cnt = len(self.controls)

        # OpenQASM use cx/ccx/c3x..., qutrunk use mcx/mcx(n)
        if name.startswith("mc"):
            name = name.replace("mc", "c", 1)

        if ctrl_cnt == 2:
            name = name.replace("c", "cc", 1)
        elif ctrl_cnt > 2:
            name = name.replace("c", "c" + str(ctrl_cnt), 1)

        angles_str = ""
        angles = self.gate.angles()
        if angles:
            angles_str = "(" + ",".join([str(ag) for ag in angles]) + ")"

        qubits_index = self.controls + self.targets
        qubits_str = ",".join([f"q[{qi}]" for qi in qubits_index])
        return name + angles_str + " " + qubits_str

    def qusl(self) -> str:
        """Generate QuSL code for command."""
        name = str(self.gate)
        params = []
        param_str = ""

        # only append control bit count as param when it's more than one
        ctrl_cnt = len(self.controls)
        if ctrl_cnt > 1:
            params.append(ctrl_cnt)

        angles = self.gate.angles()
        if angles:
            params += angles

        if params:
            param_str = "(" + ", ".join([str(param) for param in params]) + ")"

        qubits_index = self.controls + self.targets
        qubits_str = ", ".join([f"q[{qi}]" for qi in qubits_index])

        # add parentheses when qubits count is more than one
        if len(qubits_index) > 1:
            qubits_str = "(" + qubits_str + ")"

        return name + param_str + " * " + qubits_str

    @property
    def name(self) -> str:
        """Command name."""
        return self.gate

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits."""
        return len(self.controls) + len(self.targets)
