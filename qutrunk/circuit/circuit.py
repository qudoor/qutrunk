"""Circuit Module."""
import json
from typing import List, Optional, Union, Callable
import numpy as np
import uuid

from qutrunk.backends import Backend, BackendLocal
from qutrunk.circuit import CBit, CReg, Counter, QuBit, SubQureg, Qureg
from qutrunk.circuit.gates import BarrierGate, MeasureGate, PauliCoeffs, ResetGate
from qutrunk.circuit.parameter import Parameter
from qutrunk.circuit.ops import AMP
from qutrunk.exceptions import QuTrunkError
from qutrunk.tools.env_reader import backend_from_env


class QCircuit:
    """Quantum circuit.

    Args:
        backend: Used to run quantum circuits.
        name: The circuit name.
        resource: Whether enable the resource statistics function, default: False.

    Example: 
        .. code-block:: python

            from qutrunk.circuit import QCircuit
            from qutrunk.circuit.gates import H, CNOT, Measure

            # new QCircuit object
            qc = QCircuit()
            qr = qc.allocate(2)
            H * qr[0]
            CNOT * (qr[0], qr[1])
            Measure * qr[0]
            Measure * qr[1]
            res = qc.run(shots=100)
    """

    prefix = "circuit"

    def __init__(
        self,
        backend=None,
        name: Optional[str] = None,
        resource: Optional[bool] = False,
    ):
        self.qreg = None
        self.creg = None
        self.cmds = []
        self.cmd_cursor = 0
        self.counter = None

        self.qubit_indices = {}
        self.cbit_indices = {}

        # dict {Parameter: value}
        self.param_dict = {}

        # use local backend(default)
        if backend is None:
            self.backend = backend_from_env()
        else:
            if not isinstance(backend, Backend):
                raise TypeError("You supplied a backend which is not supported.\n")
            self.backend = backend

        # density not supported
        self.density = False

        self.backend.circuit = self

        if name is None:
            name = self._generate_circuit_name()
        self.name = name

        if resource:
            self.counter = Counter(self)

    def allocate(self, qubits: Union[int, list]):
        """Allocate qubit in quantum circuit.

        Args:
            qubits: int: The number of qubit allocated in circuit.\
                    list: The sum of list is the number of qubit allocated in circuit,\
                    and each value item represents the size of corresponding subqureg.

        Returns:
            The register of quantum.
        """

        if not isinstance(qubits, (int, list)):
            raise TypeError("qubits parameter should be type of int or list.")

        qubit_size = qubits if isinstance(qubits, int) else sum(qubits)
        if qubit_size <= 0:
            raise ValueError("Number of qubits should be larger than 0.")

        # if qubit_size > 25:
        #     raise ValueError("Number of qubits should be less than 25.")

        self.qreg = Qureg(circuit=self, size=qubit_size)
        self.creg = CReg(circuit=self, size=qubit_size)

        if self.counter:
            self.counter.qubits = qubit_size

        for index in range(qubit_size):
            self.qubit_indices[QuBit(self.qreg, index)] = index
            self.cbit_indices[CBit(self.creg, index)] = index

        if isinstance(qubits, int):
            return self.qreg
        elif isinstance(qubits, list):
            return self.qreg.split(qubits)

    def set_cmds(self, cmds):
        """Set cmds to circuit.

        Args:
            cmds: The commands set to circuit.
        """
        self.cmds = cmds

    def append_cmd(self, cmd):
        """Append command to circuit when apply a quantum gate.

        Args:
            cmd: The command append to circuit.
        """
        self.cmds.append(cmd)

    def append_circuit(self, circuit):
        """Append target circuit to current circuit.

        Note: The target circuit must have the same qubits as current circuit.

        Args:
            circuit: The target circuit append to current circuit.
        """
        if circuit.num_qubits != self.num_qubits:
            raise QuTrunkError(
                "The target circuit must have the same qubits as current circuit."
            )
        for cmd in circuit.cmds:
            self.append_cmd(cmd)

    def forward(self, num):
        """Update the cmd_cursor when a bunch of quantum operations have been run.

        Args:
            num: The number of cmds in current bunch.
        """
        self.cmd_cursor += num

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits.

        Returns:
            The length of the self.qreg.
        """

        return len(self.qreg)

    @property
    def num_gates(self) -> int:
        """Get the number of gates.

        Returns:
            The length of the self.cmds.
        """
        return len(self.cmds)

    def run(self, shots=1):
        """Run quantum circuit through the specified backend and shots.

        Args:
            shots: Run times of the circuit, default: 1.

        Returns:
            The Result object contain circuit running outcome.
        """
        self.backend.send_circuit(self, True)
        result = self.backend.run(shots)

        res = Result(self.backend, result, arguments={"shots": shots})

        return res

    def draw(self, output=None, line_length=None):
        """Quantum circuit.

        Draw the quantum circuit.

        Args:
            output: Set the method of drawing the circuit.
            line_length: Max length to draw quantum circuit, the excess circuit will be truncated.

        Returns:
            Visualized quantum circuit diagram.
        """
        if output is None:
            output = "text"

        from qutrunk.visualizations import circuit_drawer

        print(circuit_drawer(circuit=self, output=output, line_length=line_length))

    def __str__(self) -> str:
        return f"QCircuit({self.num_qubits})"

    def __repr__(self) -> str:
        return f"QCircuit({self.num_qubits})"

    @property
    def qubits(self) -> List[QuBit]:
        """Returns a list of quantum bits."""
        return self.qreg.qubits

    @property
    def cbits(self) -> List[CBit]:
        """Returns a list of cbit."""
        return self.creg.cbits

    def _generate_circuit_name(self):
        """Generate circuit name."""
        name = uuid.uuid4()
        return f"{self.prefix}-{name}"

    def __len__(self) -> int:
        """Return the number of operations in circuit."""
        return len(self.cmds)

    def get_prob(self, value):
        """Get probability of the possible measure result of circuit.

        Args:
            value: The target value.

        Returns:
            float: The probability of value.
        """
        if not hasattr(self.backend, "get_prob"):
            raise NotImplementedError(f"{self.backend.name} not support get_prob method.")

        self.backend.send_circuit(self)
        return self.backend.get_prob(value)

    def get_probs(self):
        """Get all probabilities of circuit.

        Returns:
            A list contains all probabilities of circuit.
        """
        if not hasattr(self.backend, "get_probs"):
            raise NotImplementedError(f"{self.backend.name} not support get_probs method.")

        qubits = [i for i in range(self.num_qubits)]
        self.backend.send_circuit(self)
        probs = self.backend.get_probs(qubits)

        out_probs = []
        for i, value in enumerate(probs):
            prob = {}
            prob["idx"] = i
            prob["prob"] = probs[i]
            out_probs.append(prob)

        return out_probs

    def _to_complex(self, state_vector):
        result = []
        for item in state_vector:
            r = item.split(",")
            result.append(complex(float(r[0]), float(r[1])))
        return result

    def get_statevector(self):
        """Get state vector of circuit."""
        if not hasattr(self.backend, "get_statevector"):
            raise NotImplementedError(f"{self.backend.name} not support get_statevector method.")

        self.backend.send_circuit(self)
        result = self._to_complex(self.backend.get_statevector())
        return np.array(result)

    def find_bit(self, bit):
        """Find locations in the circuit.

        Returns the index of the qubit or CBit in the circuit.

        Args:
            bit: QuBit or CBit.

        Returns:
            The index of QuBit or CBit in circuit.
        """
        try:
            if isinstance(bit, QuBit):
                return self.qubit_indices[bit]
            elif isinstance(bit, CBit):
                return self.cbit_indices[bit]
            else:
                raise Exception(f"Could not locate bit of unknown type:{type(bit)}")
        except KeyError:
            raise Exception(f"Could not locate provided bit:{bit}")

    def create_parameter(self, name: str):
        """
        Allocate one Parameter object.

        Args:
            name(str): Parameter name.

        Returns:
            Parameter object
        """
        p = Parameter(name)
        self.param_dict[name] = p
        return p

    def create_parameters(self, names: list) -> tuple:
        """
        Allocate a batch of parameters.

        Args:
            names(list): A list of parameter's name.

        Returns:
            tuple: a batch of parameters.
        """
        params = []
        for name in names:
            params.append(self.create_parameter(name))

        return tuple(params)

    def get_parameters(self):
        """Get all parameters of circuit.

        Args:
            list: all parameters in circuit.
        """
        param_values = [param.value for param in self.param_dict.values()]
        return param_values

    def bind_parameters(self, params):
        """
        Assign specific value to parameters.

        Args:
            params (dict): {parameter: value, ...}.

        Raises:
            ValueError: parameters variable contains parameters not present in the circuit.
        """
        if not isinstance(params, dict):
            raise ValueError("parameters must be dictionary.")
        # parameter exist or not
        parameters_table_key = self.param_dict.keys()
        params_not_in_circuit = [
            param_key
            for param_key in params.keys()
            if param_key not in parameters_table_key
        ]
        if len(params_not_in_circuit) > 0:
            raise ValueError(
                f"Cannot bind parameters ({', '.join(map(str, params_not_in_circuit))}) "
                f"not present in the circuit."
            )

        # update parameter
        for k, v in params.items():
            param = self.param_dict[k]
            param.update(v)

        # note: after binding parameters means that the circuit has changed and needs to be rebuilt
        new_circuit = QCircuit(backend=self.backend, name=self.name)
        new_circuit.allocate(qubits=self.num_qubits)
        new_circuit.set_cmds(self.cmds)
        return new_circuit

    def inverse(self):
        """Invert this circuit.

        Reverses the circuit and returns an error message.

        Returns:
            tuple: (QCircuit,qreg)

        Raises:
            ValueError: if the circuit cannot be inverted.
        """
        inverse_circuit = QCircuit(backend=self.backend, name=self.name + "_dg")
        inverse_circuit.allocate(qubits=self.num_qubits)

        # inverse cmd and gate
        cmds = self.cmds
        for cmd in reversed(cmds):
            if isinstance(cmd.gate, (MeasureGate, ResetGate, AMP)):
                raise ValueError("The circuit cannot be inverted.")
            cmd.inverse = not cmd.inverse
            inverse_circuit.append_cmd(cmd)

        return inverse_circuit, inverse_circuit.qreg

    @staticmethod
    def load(file, format=None):
        """Deserialize file object containing a OpenQASM or qusl document to a Python object.

        Args:
            file (str): Path to the file for a qusl or OpenQASM program.
            format(str): The format of file content.

        Return:
            The QCircuit object for the input qusl or OpenQASM.

        """
        if format is None or format == "qusl":
            from qutrunk.tools.qusl_parse import qusl_to_circuit

            return qusl_to_circuit(file)

        if format == "openqasm":
            from qutrunk.qasm import Qasm
            from qutrunk.converters import dag_to_circuit
            from qutrunk.converters import ast_to_dag

            qasm = Qasm(file)
            ast = qasm.parse()
            dag = ast_to_dag(ast)
            return dag_to_circuit(dag)

    def expval_pauli(self, paulis: Union[list, object]):
        """Computes the expected value of a product of Pauli operators.

        Args:
            paulis:
                oper_type (int): Pauli operators.
                target (int): indices of the target qubits.

        Returns:
            The expected value of a product of Pauli operators.
        """
        if not hasattr(self.backend, "get_expec_pauli_prod"):
            raise NotImplementedError(f"{self.backend.name} not support get_expec_pauli_prod method.")

        pauli_list = []
        if not isinstance(paulis, list):
            pauli_list.append(paulis)
        else:
            pauli_list = paulis
        self.backend.send_circuit(self)
        expect = self.backend.get_expec_pauli_prod(pauli_list)
        return expect

    def expval_pauli_sum(self, pauli_coeffs: PauliCoeffs):
        """Computes the expected value of a sum of products of Pauli operators.

        Args:
            pauli_coeffs (PauliCoeffs): Maintain a list of PauliCoeff, each PauliCoeff consist of one coefficient \
                and a series of pauli operators, PauliCoeffs is used to calculate the sum of Pauli products.

         .. note::

            The length of paulis in each term should be equal to self.num_qubits, otherwise \
                PauliI operator will be filled automatically.

        Returns:
            Returns the sum of Pauli products.

        Raises:
            ValueError: If the length of paulis in each term greater than self.num_qubits.
        """
        if not hasattr(self.backend, "get_expec_pauli_sum"):
            raise NotImplementedError(f"{self.backend.name} not support get_expec_pauli_sum method.")

        self.backend.send_circuit(self)
        paulis = []
        coeffs = []
        for term in pauli_coeffs:
            if len(term.paulis) > self.num_qubits:
                raise ValueError(
                    "The length of paulis in each term should be equal to self.num_qubits"
                )
            if len(term.paulis) < self.num_qubits:
                term.padding(self.num_qubits - len(term.paulis))
            coeffs.append(term.coeff)
            paulis.extend(term.paulis)

        return self.backend.get_expec_pauli_sum(paulis, coeffs)

    def _dump_qusl(self, file):
        with open(file, "w", encoding="utf-8") as f:
            qusl_data = {}
            qusl_data["target"] = "QuSL"
            qusl_data["version"] = "1.0"

            meta = {"circuit_name": self.name, "qubits": str(len(self.qreg))}
            qusl_data["meta"] = meta

            inst = []
            for c in self.cmds:
                inst.append(c.qusl() + "\n")

            qusl_data["code"] = inst
            f.write(json.dumps(qusl_data))

    def _dump_openqasm(self, file):
        with open(file, "w", encoding="utf-8") as f:
            f.write("OPENQASM 2.0;\n")
            f.write('include "qelib1.inc";\n')
            f.write(f"qreg q[{str(len(self.qreg))}];\n")
            f.write(f"creg c[{str(len(self.qreg))}];\n")
            for c in self.cmds:
                f.write(c.qasm() + ";\n")

    def dump(self, file=None, format=None):
        """Serialize Quantum circuit as a JSON formatted stream to file.

        Args:
            file: Dump the qutrunk instruction to file(json format).
        """
        if file is None:
            raise Exception("file argument need to be supplied.")

        if format is None or format == "qusl":
            self._dump_qusl(file)

        if format == "openqasm":
            self._dump_openqasm(file)

    def _print_qusl(self):
        """Print quantum circuit in qutrunk form."""
        print(f"qreg q[{str(len(self.qreg))}]")
        print(f"creg c[{str(len(self.qreg))}]")

        for c in self.cmds:
            print(c.qusl())

    def _print_qasm(self):
        """Print quantum circuit in OpenQASM form."""
        print("OPENQASM 2.0;")
        print('include "qulib1.inc";')
        print(f"qreg q[{str(len(self.qreg))}];")
        print(f"creg c[{str(len(self.qreg))}];")
        for c in self.cmds:
            print(c.qasm() + ";")

    def print(self, format=None):
        """Print quantum circuit.

        Args:
            format(str): The format of needed to print, Default to qusl. options are "qusl" or "openqasm".
        """
        if format is None or format == "qusl":
            self._print_qusl()

        if format == "openqasm":
            self._print_qasm()

    def depth(
        self,
        counted_gate: Optional[Callable] = lambda x: not isinstance(x, BarrierGate),
    ) -> int:
        """Return circuit depth (i.e., max length of critical path).

        Args:
            counted_gate (callable): A function to filter out some instructions.
                Should take as input a tuple of (Instruction, list(Qubit), list(CBit)).
                By default filters out barrier.

        Returns:
            int: Depth of circuit.

        """

        # If no qubits or cmds, return 0
        if not self.qreg or not self.cmds:
            return 0

        # A list that holds the depth of each qubit
        depths = [0] * len(self.qreg)

        for cmd in self.cmds:
            if not counted_gate(cmd):
                continue

            indices = cmd.controls + cmd.targets
            cur_depths = []
            for idx1 in indices:
                cur_depths.append(depths[idx1] + 1)

            max_depth = max(cur_depths)
            for idx2 in indices:
                depths[idx2] = max_depth

        return max(depths)

    def width(self) -> int:
        """Get width of circuit, namely number of QuBit plus CBit.

        Returns:
            int: Width of circuit.

        """
        return len(self.qreg) + len(self.creg)

    def show_resource(self):
        """Show resource of the circuit use."""
        if self.counter:
            self.counter.show_verbose()


class Result:
    """Result of quantum operation.

    Save the result of quantum circuit running.

    Args:
        backend: The backend that supports the operation of quantum circuits.
        res: The circuit running result from backend.
        arguments: The additional parameters, default for the expressions of shots.

    Example:
        .. code-block:: python

            from qutrunk.circuit import QCircuit
            from qutrunk.circuit.gates import H, Measure

            # new QCircuit object
            qc = QCircuit()
            qr = qc.allocate(1)
            H * qr[0]
            Measure * qr[0]
            # get running result
            res = qc.run()
            # get measurement from result
            print(res.get_measures())
            # get bitstrs
            print(res.get_bitstrs())
            # get values in decimal format
            print(res.get_values())
            # get number of each bitstr
            print(res.get_counts())
            # get running info
            print(res.running_info())
    """

    def __init__(
        self, backend, res, arguments = '{"shots": 1}'
    ):
        self.backend = backend
        self.arguments = arguments
        self.measure_result = res

    def get_measures(self, qreg: Union[Qureg, SubQureg] = None):
        """Get the measure result."""
        if not self.measure_result or not self.measure_result.measures or len(self.measure_result.measures) == 0:
            return []

        measures = []
        idxs = None
        if qreg is not None:
            idxs = qreg.get_indexs()
        column = -1
        for ms in self.measure_result.measures:
            measures.append(ms.simplify(idxs))
            column = len(measures[0])
        row = len(measures)
        return np.array(measures).reshape(row, column)

    def get_bitstrs(self, qreg: Union[Qureg, SubQureg] = None):
        """Get the measure result in binary format."""
        idxs = None
        if qreg is not None:
            idxs = qreg.get_indexs()
        return self.measure_result.get_bitstrs(idxs)

    # TODO: have some problem with this method.
    def get_values(self, qreg: Union[Qureg, SubQureg] = None):
        """Get the measure result of int."""
        idxs = None
        if qreg is not None:
            idxs = qreg.get_indexs()
        return self.measure_result.get_values(idxs)

    def get_counts(self, qreg: Union[Qureg, SubQureg] = None):
        """Get the number of times the measurement results appear."""
        if self.measure_result is None:
            return None

        res = []
        idxs = None
        if qreg is not None:
            idxs = qreg.get_indexs()
        measure_counts = self.measure_result.get_measure_counts(idxs)
        for key, value in measure_counts.items():
            res.append({key: value})
        return json.dumps(res)

    def running_info(self):
        """The resourece of run."""
        result = {
            "backend": self.backend.name,
            "task_id": self.backend.task_id,
            "status": 'success',
            "arguments": self.arguments,
        }

        return json.dumps(result)
