"""Command Module."""

from qutrunk.circuit.parameter import Parameter


class Amplitude:
    """Set state-vector Amplitude.

    Args:
        reals: Amplitude read part.
        imags: Amplitude imag part.
        startind: Amplitude start index.
        numamps: Amplitude number.
    """

    def __init__(self):
        self.reals = []
        self.imags = []
        self.startind = 0
        self.numamps = 0


class CmdEx:
    """Command extension.

    Args:
        amp: Amplitude object.
    """

    def __init__(self, amp=None):
        self.amp = amp


class Command:
    """Converts the quantum gate operation into a specific command.

    Args:
        gate: Quantum gate operator.
        targets: Target qubits.
        controls: Control qubits.
        rotation: Angle for rotation gate.
        inverse: Whether to enable the inverse circuit.
    """

    def __init__(
        self,
        gate,
        targets=None,
        controls=None,
        rotation=None,
        inverse=False,
        cmdex=None,
    ):
        # TODO: modify controls and rotation to tuple?
        if targets is None:
            self.targets = []
        else:
            self.targets = list(targets)

        if controls is None:
            self.controls = []
        else:
            self.controls = list(controls)

        if rotation is None:
            self.rotation = []
        else:
            self.rotation = list(rotation)

        self.gate = gate
        self.cmd_ver = "1.0"
        self.inverse = inverse

        self.parameters = {}
        for i, r in enumerate(self.rotation):
            if isinstance(r, Parameter):
                # map index to parameter
                self.parameters[i] = r
                r.set_host(self)

        # Command extention data
        # TODO: extra
        self.cmdex = cmdex

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
        if len(self.rotation) > 0:
            angles_str = "(" + ",".join([str(ag) for ag in self.rotation]) + ")"

        qubits_index = self.controls + self.targets
        qubits_str = ",".join([f"q[{qi}]" for qi in qubits_index])
        return name + angles_str + " " + qubits_str

    def qusl(self) -> str:
        """Generate QuSL code for command."""
        name = str(self.gate)
        if name == "AMP":
            return f"AMP({self.gate.classicvector}, {self.gate.startind}, {self.gate.numamps}) * q"
            # return 'AMP({}, {}, {}) * q'.format(self.gate.classicvector, self.gate.startind, self.gate.numamps)

        params = []
        param_str = ""
        inv_str = ""

        # only append control bit count as param when it's more than one
        ctrl_cnt = len(self.controls)
        if ctrl_cnt:
            params.append(ctrl_cnt)

        if len(self.rotation) > 0:
            params += self.rotation

        if params:
            param_str = "(" + ", ".join([str(param) for param in params]) + ")"

        qubits_index = self.controls + self.targets
        qubits_str = ", ".join([f"q[{qi}]" for qi in qubits_index])

        # add parentheses when qubits count is more than one
        if len(qubits_index) > 1:
            qubits_str = "(" + qubits_str + ")"

        if self.inverse:
            inv_str += ".inv()"

        return name + param_str + inv_str + " * " + qubits_str

    @property
    def name(self) -> str:
        """Command name."""
        return self.gate

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits."""
        return len(self.controls) + len(self.targets)

    def update_parameter(self, param):
        """Update command parameter.

        Usually, parameter is the angle of rotation gate

        Args:
            param: Parameter object holds parameter's name and value.
        """
        for k, v in self.parameters.items():
            if v == param:
                self.rotation[k] = v.value
