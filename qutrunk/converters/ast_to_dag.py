# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""AST (abstract syntax tree) to DAG (directed acyclic graph) converter.

Acts as an OpenQASM interpreter.
"""
import re
from collections import OrderedDict

from qutrunk.circuit import CReg, Qureg
from qutrunk.circuit.gates import BasicGate, MeasureGate, BarrierGate, MCX
from qutrunk.converters.mapping import qutrunk_standard_gate
from qutrunk.dagcircuit import DAGCircuit
from qutrunk.exceptions import QuTrunkError
from qutrunk.qasm.node.real import Real


def ast_to_dag(ast):
    """Build a ``DAGCircuit`` object from an AST ``Node`` object.

    Args:
        Ast (Program): A Program Node of an AST (parser's output).

    Return:
        DAGCircuit: The DAG representing an OpenQASM's AST.

    Raises:
        QuTrunkError: If the AST is malformed.
    """
    dag = DAGCircuit()
    AstInterpreter(dag)._process_node(ast)

    return dag


class AstInterpreter:
    """Interprets an OpenQASM by expanding subroutines and unrolling loops.

    Args:
        dag: The DAG representing an OpenQASM's AST.
    """

    def __init__(self, dag):
        """Initialize interpreter's data."""
        # DAG object to populate
        self.dag = dag
        # OPENQASM version number (ignored for now)
        self.version = 0.0
        # Dict of gates names and properties
        self.gates = OrderedDict()
        # Keeping track of conditional gates
        self.condition = None
        # List of dictionaries mapping local parameter ids to expression Nodes
        self.arg_stack = [{}]
        # List of dictionaries mapping local bit ids to global ids (name, idx)
        self.bit_stack = [{}]

    def _process_bit_id(self, node):
        """Process an Id or IndexedId node as a bit or register type.

        Returns:
            Return a list of tuples (Register,index).

        Raises:
            QuTrunkError: If the AST is malformed.
        """
        reg = None

        if node.name in self.dag.qregs:
            reg = self.dag.qregs[node.name]
        elif node.name in self.dag.cregs:
            reg = self.dag.cregs[node.name]
        else:
            raise QuTrunkError(
                "expected qreg or creg name:",
                "line=%s" % node.line,
                "file=%s" % node.file,
            )

        if node.type == "indexed_id":
            # An indexed bit or qubit
            return [reg[node.index]]
        elif node.type == "id":
            # A qubit or qreg or creg
            if not self.bit_stack[-1]:
                # Global scope
                return list(reg)
            else:
                # local scope
                if node.name in self.bit_stack[-1]:
                    return [self.bit_stack[-1][node.name]]
                raise QuTrunkError(
                    "expected local bit name:",
                    "line=%s" % node.line,
                    "file=%s" % node.file,
                )
        return None

    def _process_custom_unitary(self, node):
        """Process a custom unitary node.

        Raises:
            QuTrunkError: If the AST is malformed.
        """
        name = node.name
        if node.arguments is not None:
            args = self._process_node(node.arguments)
        else:
            args = []
        bits = [
            self._process_bit_id(node_element) for node_element in node.bitlist.children
        ]

        if name in self.gates:
            self._arguments(name, bits, args)
        else:
            raise QuTrunkError(
                "internal error undefined gate:",
                "line=%s" % node.line,
                "file=%s" % node.file,
            )

    def _process_u(self, node):
        """Process a U gate node."""
        args = self._process_node(node.arguments)
        bits = [self._process_bit_id(node.bitlist)]

        self._arguments("u", bits, args)

    def _arguments(self, name, bits, args):
        """Gate arguments."""
        gargs = self.gates[name]["args"]
        gbits = self.gates[name]["bits"]

        maxidx = max(map(len, bits))
        for idx in range(maxidx):
            self.arg_stack.append({gargs[j]: args[j] for j in range(len(gargs))})
            # Only index into register arguments.
            element = [idx * x for x in [len(bits[j]) > 1 for j in range(len(bits))]]
            self.bit_stack.append(
                {gbits[j]: bits[j][element[j]] for j in range(len(gbits))}
            )
            self._create_dag_op(
                name,
                [self.arg_stack[-1][s].sym() for s in gargs],
                [self.bit_stack[-1][s] for s in gbits],
            )
            self.arg_stack.pop()
            self.bit_stack.pop()

    def _process_gate(self, node, opaque=False):
        """Process a gate node.

        If opaque is True, process the node as an opaque gate node.
        """
        self.gates[node.name] = {}
        de_gate = self.gates[node.name]
        de_gate["print"] = True  # default
        de_gate["opaque"] = opaque
        de_gate["n_args"] = node.n_args()
        de_gate["n_bits"] = node.n_bits()
        if node.n_args() > 0:
            de_gate["args"] = [element.name for element in node.arguments.children]
        else:
            de_gate["args"] = []
        de_gate["bits"] = [c.name for c in node.bitlist.children]
        if node.name in qutrunk_standard_gate:
            return
        if opaque:
            de_gate["body"] = None
        else:
            de_gate["body"] = node.body

    def _process_cnot(self, node):
        """Process a CNOT gate node.

        Raises:
            QuTrunkError: If the AST is malformed.
        """
        id0 = self._process_bit_id(node.children[0])
        id1 = self._process_bit_id(node.children[1])
        if not (len(id0) == len(id1) or len(id0) == 1 or len(id1) == 1):
            raise QuTrunkError(
                "internal error: qreg size mismatch",
                "line=%s" % node.line,
                "file=%s" % node.file,
            )
        maxidx = max([len(id0), len(id1)])
        for idx in range(maxidx):
            cx_gate = MCX()
            if len(id0) > 1 and len(id1) > 1:
                self.dag.apply_operation_back(cx_gate, [id0[idx], id1[idx]], [])
            elif len(id0) > 1:
                self.dag.apply_operation_back(cx_gate, [id0[idx], id1[0]], [])
            else:
                self.dag.apply_operation_back(cx_gate, [id0[0], id1[idx]], [])

    def _process_measure(self, node):
        """Process a measurement node."""
        id0 = self._process_bit_id(node.children[0])
        id1 = self._process_bit_id(node.children[1])
        if len(id0) != len(id1):
            raise QuTrunkError(
                "internal error: reg size mismatch",
                "line=%s" % node.line,
                "file=%s" % node.file,
            )
        for idx, idy in zip(id0, id1):
            meas_gate = MeasureGate()
            meas_gate.name = "measure"
            self.dag.apply_operation_back(meas_gate, [idx], [idy])

    # note: qutrunk not support if condition
    def _process_if(self, node):
        """Process an if node."""
        pass

    def _process_children(self, node):
        """Call process_node for all children of node."""
        for kid in node.children:
            self._process_node(kid)

    def _process_node(self, node):
        """Carry out the action associated with a node.

        Raises:
            QuTrunkError: If the AST is malformed.
        """
        if node.type == "program":
            self._process_children(node)

        elif node.type == "qreg":
            qreg = Qureg(size=node.index, name=node.name)
            self.dag.add_qreg(qreg)

        elif node.type == "creg":
            creg = CReg(size=node.index, name=node.name)
            self.dag.add_creg(creg)

        elif node.type == "id":
            raise QuTrunkError("internal error: _process_node on id")

        elif node.type == "int":
            raise QuTrunkError("internal error: _process_node on int")

        elif node.type == "real":
            raise QuTrunkError("internal error: _process_node on real")

        elif node.type == "indexed_id":
            raise QuTrunkError("internal error: _process_node on indexed_id")

        elif node.type == "id_list":
            # We process id_list nodes when they are leaves of barriers.
            return [
                self._process_bit_id(node_children) for node_children in node.children
            ]

        elif node.type == "primary_list":
            # We should only be called for a barrier.
            return [self._process_bit_id(m) for m in node.children]

        elif node.type == "gate":
            self._process_gate(node)

        elif node.type == "custom_unitary":
            self._process_custom_unitary(node)

        elif node.type == "universal_unitary":
            self._process_u(node)

        elif node.type == "cnot":
            self._process_cnot(node)

        elif node.type == "expression_list":
            return node.children

        elif node.type == "binop":
            raise QuTrunkError("internal error: _process_node on binop")

        elif node.type == "prefix":
            raise QuTrunkError("internal error: _process_node on prefix")

        elif node.type == "measure":
            self._process_measure(node)

        elif node.type == "format":
            self.version = node.version()

        elif node.type == "barrier":
            ids = self._process_node(node.children[0])
            qubits = []
            for qubit in ids:
                for j, _ in enumerate(qubit):
                    qubits.append(qubit[j])
            self.dag.apply_operation_back(BarrierGate(), qubits, [])

        elif node.type == "reset":
            id0 = self._process_bit_id(node.children[0])
            for i, _ in enumerate(id0):
                # qutrunk not support reset yet
                pass

        elif node.type == "if":
            self._process_if(node)

        elif node.type == "opaque":
            self._process_gate(node, opaque=True)

        elif node.type == "external":
            raise QuTrunkError("internal error: _process_node on external")

        else:
            raise QuTrunkError(
                "internal error: undefined node type",
                node.type,
                "line=%s" % node.line,
                "file=%s" % node.file,
            )
        return None

    def _gate_rules_to_qutrunk_circuit(self, node, params):
        """From a gate definition in qasm, to a QCircuit format."""
        rules = []
        qreg = Qureg(size=node["n_bits"])
        bit_args = {node["bits"][i]: q for i, q in enumerate(qreg)}
        exp_args = {node["args"][i]: Real(q) for i, q in enumerate(params)}

        for child_op in node["body"].children:
            qparams = []
            eparams = []
            for param_list in child_op.children[1:]:
                if param_list.type == "id_list":
                    qparams = [bit_args[param.name] for param in param_list.children]
                elif param_list.type == "expression_list":
                    for param in param_list.children:
                        eparams.append(param.sym(nested_scope=[exp_args]))
            op = self._create_op(child_op.name, params=eparams)
            rules.append((op, qparams, []))

        sub_circ = []
        for instr, qargs, cargs in rules:
            sub_circ.append((instr, qargs, cargs))
        return sub_circ

    def _create_dag_op(self, name, params, qargs):
        """Create a DAG node out of a parsed AST op node.

        Args:
            name (str): Operation name to apply to the DAG.
            params (list): Op parameters.
            qargs (list(Qubit)): Qubits to attach to.

        Raises:
            QuTrunkError: If encountering a non-basis opaque gate.
        """
        op = self._create_op(name, params)
        self.dag.apply_operation_back(op, qargs, [])

    def _create_op(self, name, params):
        # check whether gate name indicate multi control bit
        name, ctr_cnt = count_ctrl(name)
        if ctr_cnt:
            params = [ctr_cnt] + params

        if name in qutrunk_standard_gate:
            op = qutrunk_standard_gate[name](*params)
            op.name = name
        elif name in self.gates:
            op = BasicGate(*params)
            op.name = name
            if not self.gates[name]["opaque"]:
                # call a custom gate (otherwise, opaque)
                op.definition = self._gate_rules_to_qutrunk_circuit(
                    self.gates[name], params=params
                )
        else:
            raise QuTrunkError("unknown operation for ast node name %s" % name)
        return op


def count_ctrl(name):
    """
    Check whether gate name indicate multi control bit
    if yes, replace cc/c[0-9]+ with 'mc' and get control bits count
    control bits always at the beginning in qargs.

    Args:
        name (str): Origin operation name in OpenQASM.

    Return:
        name (str): Operation name after process.
        cnt (int): Control bits count.
    """
    cnt = 0
    multi_ctrl_reg = r"c(c|\d+)"
    match = re.match(multi_ctrl_reg, name)
    if match:
        cnt_str = match.group(1)
        if cnt_str == "c":
            cnt = 2
        else:
            cnt = int(cnt_str)
        name = re.sub(multi_ctrl_reg, "mc", name)
    return name, cnt
