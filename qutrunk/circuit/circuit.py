"""Circuit Module."""
import json
import random
from typing import List, Optional, Union

from qutrunk.backends import Backend, BackendLocal
from qutrunk.circuit import CBit, CReg, Counter, QuBit, Qureg
from qutrunk.circuit.gates import BarrierGate, MeasureGate, Observable
from qutrunk.circuit.parameter import Parameter
from qutrunk.circuit.ops import AMP


class QCircuit:
    """Quantum circuit.

    Args:
        backend: Used to run quantum circuits.
        density: Creates a density matrix Qureg object representing a set of \
            qubits which can enter noisy and mixed states.
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
        density=False,
        name: Optional[str] = None,
        resource: Optional[bool] = False,
    ):
        self.qreg = None
        self.creg = None
        # record the original statement: gate or operator
        self.statements = []
        self.cmds = []
        self.cmd_cursor = 0
        self.counter = None
        # mark circuit in Operator Context
        self._in_op = False

        self.qubit_indices = {}
        self.cbit_indices = {}

        # 参数字典�{Parameter: value}
        self.parameters = {}

        # use local backend(default)
        if backend is None:
            self.backend = BackendLocal()
        else:
            if not isinstance(backend, Backend):
                raise TypeError("You supplied a backend which is not supported.\n")
            self.backend = backend

        # local backend is no support noisy pattern.
        if density and isinstance(backend, BackendLocal):
            raise TypeError("You supplied a backend which is not supported density.\n")
        self.density = density

        self.backend.circuit = self
        # TODO:?
        self.outcome = None

        if name is None:
            name = self._generate_circuit_name()
        self.name = name

        if resource:
            self.counter = Counter(self)

    def __iter__(self):
        """Used to iterate commands in quantum circuits."""
        return QCircuitIter(self.cmds)

    def allocate(self, qubits: Union[int, list]):
        """Allocate qubit in quantum circuit.

        Args:
            qubits: int: The number of qubit allocated in circuit.\
                    list: The sum of list is the number of qubit allocated in circuit,\
                    and each value item represents the size of corresponding subqureg.

        Returns:
            # TODO: update description and demo
            qreg: The register of quantum.
        """
        if not isinstance(qubits, (int, list)):
            raise TypeError("qubits parameter should be type of int or list.")

        qubit_size = qubits if isinstance(qubits, int) else sum(qubits)
        if qubit_size <= 0:
            raise TypeError("Number of qubits should be larger than 0.")

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

    def append_statement(self, statement):
        """Append the origin statement when apply gate or operator.

        Args:
            statement: The statement when apply gate or operator.
        """
        self.statements.append(statement)

    # TODO: need to improve.
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

    def set_measure(self, qubit, value):
        """Store the measure result for target qubit.

        Args:
            qubit: The index of qubit in qureg.
            value: The qubit measure value(0 or 1).

        Raises:
            IndexError: Qubit index must be less than then length of qreg.
        """
        if qubit >= len(self.qreg) or qubit < 0:
            raise IndexError("qubit index out of range.")
        self.creg[qubit].value = value

    # TODO: need to improve.
    def run(self, shots=1):
        """Run quantum circuit through the specified backend and shots.

        Args:
            shots: Run times of the circuit, default: 1.

        Returns:
            result: The Result object contain circuit running outcome.
        """
        self.backend.send_circuit(self, True)
        result = self.backend.run(shots)

        if self.backend.name == "BackendIBM":
            # note: ibm后端运行结果和qutrunk差异较大，目前直接将结果返回不做适配
            return result
        # TODO: measureSet
        if result and result.measureSet:
            for m in result.measureSet:
                self.set_measure(m.id, m.value)

        res = Result(self.num_qubits, result, self.backend, arguments={"shots": shots})

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
        return str(self.draw(output="text"))

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
        return f"{self.prefix}-{str(random.getrandbits(15))}"

    def __len__(self) -> int:
        """Return the number of operations in circuit."""
        return len(self.cmds)

    def get_prob_value(self, value):
        """Probability of obtaining quantum circuit measurements.

        Args:
            value: The target value.

        Returns:
            float: The probability of value.
        """
        self.backend.send_circuit(self)
        return self.backend.get_prob_amp(value)

    def get_prob_qubit_value(self, qubit, value):
        """Get the probability of a specified qubit being measured in the given outcome (0 or 1).

        Args:
            qubit: The specified qubit to be measured.
            value: The qubit measure result(0 or 1).

        Returns:
            The probability of target qubit.
        """
        if qubit < 0 or qubit >= self.num_qubits:
            raise IndexError("out of the range of qubits.")

        self.backend.send_circuit(self)
        return self.backend.get_prob_outcome(qubit, value)

    # TODO:Get the maximum possible value of a qubit.
    def get_prob_qubits(self, qubits=None):
        """Get outcomeProbs with the probabilities of every outcome of the sub-register contained in qureg.

        Args:
            qubits: The sub-register contained in qureg.

        Returns:
            An array contains probability of target qubits.
        """
        if qubits is None:
            qubits = [i for i in range(self.num_qubits)]
        else:
            if not all(isinstance(qubit, int) for qubit in qubits):
                raise TypeError("The argument must be integer.")

        self.backend.send_circuit(self)
        return self.backend.get_prob_all_outcome(qubits)

    # TODO: need to improve.
    def get_all_state(self):
        """Get the current state vector of probability amplitudes for a set of qubits."""
        self.backend.send_circuit(self)
        return self.backend.get_all_state()

    def find_bit(self, bit):
        """Find locations in the circuit.

        Returns the index of the qubit or CBit in the circuit.

        Args:
            bit: QuBit or CBit.

        Returns:
            index: The index of QuBit or CBit in circuit.
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

    def parameter(self, name):
        """
        get a new object of Parameter.

        Args:
            name(str): Parameter name.

        Returns:
            p: Parameter object
        """
        p = Parameter(name)
        self.parameters[name] = p
        return p

    def get_parameter(self, name):
        """get the object of Parameter.

        Args:
            name(str): Parameter name.
        """
        return self.parameters[name]

    def bind_parameters(self, params):
        """
        Assign numeric parameters to parameters.

        Args:
            params (dict): {parameter: value, ...}.

        Raises:
            ValueError: parameters variable contains parameters not present in the circuit.

        Returns:
            QCircuit:Quantum circuit
        """
        if not isinstance(params, dict):
            raise ValueError("parameters must be dictionary.")
        # 1 参数是否在参数表�
        parameters_table_key = self.parameters.keys()
        params_not_in_circuit = [
            param_key
            for param_key in params.keys()
            if param_key not in parameters_table_key
        ]
        if len(params_not_in_circuit) > 0:
            raise ValueError(f"Cannot bind parameters ({', '.join(map(str, params_not_in_circuit))}) "
                             f"not present in the circuit.")

        # update parameter
        for k, v in params.items():
            param = self.parameters[k]
            param.update(v)

    def get_parameter_value(self, name):
        """get the value of Parameter.

        Args:
            name(str): Parameter name.
        """
        for k, v in self.parameters.items():
            if name == k:
                return v.value

        return None

    def inverse(self):
        """Invert this circuit.

        Reverses the circuit and returns an error message.

        Returns:
            QCircuit: The inverted circuit.
            qreg: The register of quantum.

        Raises:
            ValueError: if the circuit cannot be inverted.
        """
        inverse_circuit = QCircuit(backend=self.backend, name=self.name + "_dg")
        inverse_circuit.allocate(qubits=self.num_qubits)

        # inverse cmd and gate
        cmds = self.cmds
        for cmd in reversed(cmds):
            if isinstance(cmd.gate, (MeasureGate, AMP)):
                raise ValueError("The circuit cannot be inverted.")
            cmd.inverse = True
            inverse_circuit.append_cmd(cmd)

        return inverse_circuit, inverse_circuit.qreg

    @staticmethod
    def load(file, format=None):
        """Deserialize file object containing a OpenQASM or qusl document to a Python object.

        Args:
            file (str): Path to the file for a qusl or OpenQASM program.
            format(str): The format of file content.

        Return:
            QCircuit: The QCircuit object for the input qusl or OpenQASM.

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

    # TODO: need to improve.
    def expval(self, obs_data):
        """Computes the expected value of a product of Pauli operators.

        Args:
            pauliprodlist:
                oper_type (int): Pauli operators.
                target (int): indices of the target qubits.

        Returns:
            The expected value of a product of Pauli operators.
        """
        self.backend.send_circuit(self)
        expect = self.backend.get_expec_pauli_prod(obs_data)
        return expect

    # TODO: need to improve.
    def expval_sum(self, pauli_coeffi: Observable, qubitnum=0):
        """Computes the expected value of a sum of products of Pauli operators.

        Args:
            pauliprodlist:
                oper_type (int): Pauli operators.
                term_coeff (float): The coefficients of each term in the sum of Pauli products.

        Returns:
            Returns the operation type and the coefficients of each term in the sum of Pauli products.
        """
        self.backend.send_circuit(self)
        pauli_type_list, coeffi_list = pauli_coeffi.obs_data()
        if (qubitnum != 0) and (len(coeffi_list) * qubitnum) != len(pauli_type_list):
            raise AttributeError(
                "Parameter error: The number of parameters is not correct."
            )
        return self.backend.get_expec_pauli_sum(pauli_type_list, coeffi_list)

    def _dump_qusl(self, file, unroll=True):
        with open(file, "w", encoding="utf-8") as f:
            qusl_data = {}
            qusl_data["target"] = "QuSL"
            qusl_data["version"] = "1.0"

            meta = {"circuit_name": self.name, "qubits": str(len(self.qreg))}
            qusl_data["meta"] = meta

            inst = []
            if unroll:
                for c in self:
                    inst.append(c.qusl() + "\n")
            else:
                for stm in self.statements:
                    inst.append(stm + "\n")

            qusl_data["code"] = inst
            f.write(json.dumps(qusl_data))

    def _dump_openqasm(self, file):
        with open(file, "w", encoding="utf-8") as f:
            f.write("OPENQASM 2.0;\n")
            f.write('include "qulib1.inc";\n')
            f.write(f"qreg q[{str(len(self.qreg))}];\n")
            f.write(f"creg c[{str(len(self.qreg))}];\n")
            for c in self:
                f.write(c.qasm() + ";\n")

    def dump(self, file=None, format=None, unroll=True):
        """Serialize Quantum circuit as a JSON formatted stream to file.

        Args:
            unroll: True: Dump the detailed instructions, especially, \
                if the instruction contains an operator, the operator will be expanded.
                False: Dump the brief instructions, the operator will not be expanded.
            file: Dump the qutrunk instruction to file(json format).
        """
        if file is None:
            raise Exception("file argument need to be supplied.")

        if format is None or format == "qusl":
            self._dump_qusl(file, unroll)

        if format == "openqasm":
            self._dump_openqasm(file)

    def _print_qusl(self, unroll):
        """Print quantum circuit in qutrunk form.

        Args:
            unroll: True: Dump the detailed instructions, especially, \
                if the instruction contains an operator, the operator will be expanded.
                False: Dump the brief instructions, the operator will not be expanded.
        """
        print(f"qreg q[{str(len(self.qreg))}]")
        print(f"creg c[{str(len(self.qreg))}]")
        if unroll:
            for c in self:
                print(c.qusl())
        else:
            for stm in self.statements:
                print(stm)

    def _print_qasm(self):
        """Print quantum circuit in OpenQASM form."""
        print("OPENQASM 2.0;")
        print('include "qulib1.inc";')
        print(f"qreg q[{str(len(self.qreg))}];")
        print(f"creg c[{str(len(self.qreg))}];")
        for c in self:
            print(c.qasm() + ";")

    def print(self, format=None, unroll=True):
        """Print quantum circuit.

        Args:
            format(str): The format of needed to print, Default to qusl. options are "qusl" or "openqasm".
            unroll: True: Dump the detailed instructions, especially, \
                if the instruction contains an operator, the operator will be expanded.
                False: Dump the brief instructions, the operator will not be expanded.
        """
        if format is None or format == "qusl":
            self._print_qusl(unroll)

        if format == "openqasm":
            self._print_qasm()

    # TODO: need to improve.
    def depth(
        self,
        counted_gate: Optional[callable] = lambda x: not isinstance(x, BarrierGate),
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

    # TODO: need to improve.
    def enter_op(self):
        """Mark circuit in Operator Context."""
        self._in_op = True

    # TODO: need to improve.
    def exit_op(self):
        """Mark circuit out Operator Context."""
        self._in_op = False

    # TODO: need to improve.
    def in_op(self):
        """Get circuit Operator Context."""
        return self._in_op


class QCircuitIter:
    """The iterator for circuit.

    Args:
       cmds: Commands to iterate.
    """

    def __init__(self, cmds):
        self.idx = 0
        self.__cmds = cmds

    def __iter__(self):
        return self

    def __next__(self):
        try:
            cmd = self.__cmds[self.idx]
        except IndexError:
            raise StopIteration

        self.idx += 1
        return cmd


class Result:
    """Result of quantum operation.

    Save the result of quantum circuit running.

    Args:
        num_qubits: The number of qubits.
        res: The circuit running result from backend.
        backend: The backend that supports the operation of quantum circuits.
        task_id: Task id will automatic generate when submit a quantum computing job.
        status: The operating state of a quantum circuit.

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
            print(res.get_measure())
    """

    def __init__(
        self, num_qubits, res, backend, arguments, task_id=None, status="success"
    ):
        self.states = []
        self.values = []
        self.backend = backend
        self.task_id = task_id
        self.status = status
        self.arguments = arguments
        # Modified quantum gate annotation.
        self.measure_result = [-1] * num_qubits

        if res and res.measureSet:
            for m in res.measureSet:
                self.set_measure(m.id, m.value)

        # Count the number of occurrences of the bit-bit string composed of all qubits.
        if res and res.outcomeSet:
            self.outcome = res.outcomeSet
            for out in self.outcome:
                # 二进制字符串转换成十进制
                if out.bitstr:
                    self.states.append(int(out.bitstr, base=2))
                    self.values.append(out.count)
        else:
            self.outcome = None

    def set_measure(self, qubit, value):
        """Update qubit measure value.

        Args:
            qubit: The index of qubit in qureg.
            value: The qubit measure value(0 or 1).
        """
        if qubit >= len(self.measure_result) or qubit < 0:
            raise IndexError("qubit index out of range.")
        self.measure_result[qubit] = value

    def get_measure(self):
        """Get the measure result."""
        return self.measure_result

    def get_outcome(self):
        """Get the measure result in binary format."""
        out = self.measure_result[::-1]
        # TODO:improve
        bit_str = "0b"
        for o in out:
            bit_str += str(o)
        return bit_str

    def get_counts(self):
        """Get the number of times the measurement results appear."""
        # TODO:improve
        if self.outcome is None:
            return None

        res = []
        for out in self.outcome:
            res.append({out.bitstr: out.count})
        return json.dumps(res)

    def get_states(self):
        """Get all states"""
        return self.states

    def get_values(self):
        """Get all values"""
        return self.values
        
    def excute_info(self):
        result = {
            "backend": self.backend.name,
            "task_id": self.task_id,
            "status": self.status,
            "arguments": self.arguments,
        }
        return json.dumps(result)
