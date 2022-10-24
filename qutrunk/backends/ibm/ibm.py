"""IBM Backend"""

import math

from qutrunk.backends import Backend
from qutrunk.circuit.gates import CNOT, H, Measure, Rx, Ry, Rz
from .ibm_client import send


# TODO:The IBM quantum chip can only do U1,U2,U3,barriers, and CX / CNOT.
class BackendIBM(Backend):
    """IBM Backend.

    The BackendIBM class, which stores the circuit, transforms it to JSON,
    and sends the circuit through the IBM API.

    Args:
        token: Token used for authorization when send quantum circuit to IBM Backend.
        device: Device type, default: ibmq_qasm_simulator.

    Example:
        .. code-block:: python

            from qutrunk.circuit import QCircuit
            from qutrunk.backends import BackendIBM
            from qutrunk.circuit.gates import H, CNOT, Measure

            # use BackendIBM
            token = "IBM Quantum token"
            qc = QCircuit(backend=BackendIBM(token=token))
            qr = qc.allocate(2)

            # apply gate
            H * qr[0]
            CNOT * (qr[0], qr[1])
            Measure * qr[0]
            Measure * qr[1]

            # run circuit
            res = qc.run(shots=100)
            print(res)
    """

    def __init__(self, token, device=None):
        super().__init__()
        self.circuit = None
        self._token = token
        # default device: "ibmq_qasm_simulator"
        self.device = "ibmq_qasm_simulator"
        # the qubits to allocated
        self._allocated_qubits = set()
        # quantum circuit for json format
        self._json = []
        # measured qubit id
        self._measured_ids = []

    def send_circuit(self, circuit, final=False):
        """Send the quantum circuit to IBM backend.

        Args:
            circuit: Quantum circuit to send.
            final: True if quantum circuit finish, default False.

        Returns:
            The result return from IBM Backend.
        """
        self._allocated_qubits.add(len(circuit.qreg))

        for ct in circuit:
            self._circuit_to_json(ct)

        for measured_id in self._measured_ids:
            self._json.append(
                {"qubits": [measured_id], "name": "measure", "memory": [measured_id]}
            )
        # print("send circuit:", self._json)

    def _circuit_to_json(self, cmd):
        """Translates the command and in a local variable.

        Args:
            cmd: The command convert to json format.
        """
        gate = cmd.gate

        if gate is Measure:
            self._measured_ids += cmd.targets
        elif gate is CNOT and len(cmd.controls) == 1:
            self._json.append({"qubits": [*cmd.controls, *cmd.targets], "name": "cx"})
        elif gate is H:
            self._json.append(
                {"qubits": cmd.targets, "name": "u2", "params": [0, 3.141592653589793]}
            )
        elif isinstance(gate, (Rx, Ry, Rz)):
            u_name = {"Rx": "u3", "Ry": "u3", "Rz": "u1"}
            u_angle = {
                "Rx": [gate.angle, -math.pi / 2, math.pi / 2],
                "Ry": [gate.angle, 0, 0],
                "Rz": [gate.angle],
            }

            gate_name = u_name[str(gate)[0:2]]
            params = u_angle[str(gate)[0:2]]

            self._json.append(
                {"qubits": cmd.targets, "name": gate_name, "params": params}
            )

        else:
            raise Exception(
                "This command is not currently supported.\n"
                + "The IBM quantum chip can only do U1,U2,U3,barriers, and CX / CNOT."
            )

    def run(self, shots=1):
        info = {}
        info["json"] = self._json

        info["nq"] = sum(self._allocated_qubits)
        info["shots"] = shots
        info["maxCredits"] = 10
        info["backend"] = {"name": self.device}

        result = send(
            info,
            device=self.device,
            token=self._token,
            num_retries=shots,
            verbose=True,
        )
        return result

    @property
    def name(self):
        return "BackendIBM"
