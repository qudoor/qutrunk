import sys
from warnings import warn
import shutil

from qutrunk.circuit import CBit
from qutrunk.circuit.gates import (
    MeasureGate,
    HGate,
    XGate,
    BarrierGate,
    MCZ,
    MCX,
    P,
    ZGate,
    R,
    Rx,
    Ry,
    Rz,
    SGate,
    SdgGate,
    TGate,
    TdgGate,
    YGate,
    SwapGate,
    SqrtXGate,
    CYGate,
    CRx,
    CRy,
    CRz,
    Rxx,
    Ryy,
    Rzz,
    U1,
    U2,
    U3,
    SqrtSwapGate,
    CP,
    X1Gate,
    Y1Gate,
    Z1Gate,
    CU,
    CU1,
    CU3,
    CR,
    iSwapGate,
    IGate,
    CHGate,
    CSwapGate,
    SqrtXdgGate,
    CSqrtXGate,
    Matrix,
    ResetGate
)
from .text_draw_element import InputWire
from .layer import Layer
from .text_draw_element import MeasureFrom, MeasureTo
from .text_draw_element import BoxOnQuWire
from .text_draw_element import Bullet, OpenBullet
from .text_draw_element import EmptyWire, BreakWire
from .text_draw_element import Barrier as BarrierBox
from .text_draw_element import Ex, iEx


class TextDrawing:
    """The text drawing.

    Args:
        qubits: Qubits.
        cbits: Classical bits.
        nodes: Operator nodes.
        circuit: Quantum circuit.
        plot_barriers: Wheter to draw barriers.
        line_length: Draw width.
        vertical_compression: Vertical compression type, available values: ["high", "medium", "low"].
        encoding: Encoding Type, default: UTF-8.

    Raises:
          ValueError.
    """

    def __init__(
        self,
        qubits,
        cbits,
        nodes,
        circuit=None,
        plot_barriers=True,
        line_length=None,
        vertical_compression="medium",
        encoding=None,
    ):
        # quantum qubits
        self.qubits = qubits
        # classics cbits
        self.cbits = cbits
        # operation node
        self.nodes = nodes
        # print barriers or not, default is true
        self.plot_barriers = plot_barriers
        # max line_length
        self.line_length = line_length
        # quantum circuit
        self.circuit = circuit

        # vertically compressed square matrix in text mode
        if vertical_compression not in ["high", "medium", "low"]:
            raise ValueError(
                "Vertical compression can only be 'high', 'medium', or 'low'"
            )
        self.vertical_compression = vertical_compression

        # encoding format
        if encoding:
            self.encoding = encoding
        else:
            if sys.stdout.encoding:
                self.encoding = sys.stdout.encoding
            else:
                self.encoding = "utf8"

    def __str__(self):
        return self.single_string()

    def __repr__(self):
        return self.single_string()

    def single_string(self):
        """Create a long string with the ascii art."""
        try:
            return "\n".join(self.lines()).encode(self.encoding).decode(self.encoding)
        except (UnicodeEncodeError, UnicodeDecodeError):
            warn(
                f"The encoding {self.encoding} has a limited charset. Consider a different encoding in your "
                "environment. UTF-8 is being used instead.",
                RuntimeWarning,
            )
            self.encoding = "utf-8"

            return "\n".join(self.lines()).encode(self.encoding).decode(self.encoding)

    def lines(self, line_length=None):
        """Generates a list with lines."""
        if line_length is None:
            line_length = self.line_length

        if not line_length:
            line_length, _ = shutil.get_terminal_size()

        # quantum num_qubits
        num_qubits = len(self.qubits)

        # TODO:build layers
        layers = self.build_layers()

        layer_groups = [[]]
        rest_of_the_line = line_length
        for layerno, layer in enumerate(layers):
            layers[layerno] = EmptyWire.fillup_layer(layer, num_qubits)

            TextDrawing.normalize_width(layer)

            if line_length == -1:
                layer_groups[-1].append(layer)
                continue

            # chop the layer to the line_length (pager)
            layer_length = layers[layerno][0].length

            if layer_length < rest_of_the_line:
                layer_groups[-1].append(layer)
                rest_of_the_line -= layer_length
            else:
                layer_groups[-1].append(BreakWire.fillup_layer(len(layer), "»"))

                # New group
                layer_groups.append([BreakWire.fillup_layer(len(layer), "«")])
                rest_of_the_line = line_length - layer_groups[-1][-1][0].length

                layer_groups[-1].append(
                    InputWire.fillup_layer(self.wire_names(with_initial_state=False))
                )
                rest_of_the_line -= layer_groups[-1][-1][0].length

                layer_groups[-1].append(layer)
                rest_of_the_line -= layer_groups[-1][-1][0].length

        lines = []
        for layer_group in layer_groups:
            wires = list(zip(*layer_group))
            lines += self.draw_wires(wires)

        # TODO:compression in layer
        # -| H |----------
        # --------| H |---
        # compression to:
        # -| H |---
        # -| H |---
        return lines

    # TODO:doing
    def node_to_gate(self, node, layer):
        """Turn the operation node into a corresponding gate operation \
            and connect the corresponding qubit bit.
        """
        # implement h, cx, measure gate
        op = node.op  # Command object
        current_cons = []
        connection_label = None
        # condition control
        conditional = False
        
        condstr = ""
        cond = op.measurecond
        if cond and cond.enable == True:
            condstr = "(q[" + str(cond.idx) + "]==" + str(cond.cond_value) + ")"

        def add_connected_gate(node, gates, layer, current_cons):
            for i, gate in enumerate(gates):
                actual_index = self.qubits.index(node.qargs[i])
                if actual_index not in [i for i, j in current_cons]:
                    layer.set_qubit(node.qargs[i], gate)
                    current_cons.append((actual_index, gate))

        # Measure
        if isinstance(op.gate, MeasureGate):
            gate = MeasureFrom()
            layer.set_qubit(node.qargs[0], gate)
            register, _, reg_index = get_bit_reg_index(self.circuit, node.cargs[0])
            if register is not None:
                layer.set_cbit(
                    node.cargs[0],
                    MeasureTo(str(reg_index)),
                )
            else:
                layer.set_clbit(node.cargs[0], MeasureTo())
        # H gate
        elif isinstance(op.gate, HGate):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("H", conditional=conditional))
        # CH gate
        elif isinstance(op.gate, CHGate):
            ctrl_text = None
            params_array = TextDrawing.controlled_wires(node, layer)
            controlled_top, controlled_bot, controlled_edge, rest = params_array

            gates = self.set_ctrl_state(
                node, conditional, ctrl_text, bool(controlled_bot)
            )
            gates.append(BoxOnQuWire("CH", conditional=conditional))
            add_connected_gate(node, gates, layer, current_cons)
        # I gate
        elif isinstance(op.gate, IGate):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("I", conditional=conditional))
        # CX gate
        elif isinstance(op.gate, MCX) and op.gate.ctrl_cnt == 1:
            ctrl_text = None
            params_array = TextDrawing.controlled_wires(node, layer)
            controlled_top, controlled_bot, controlled_edge, rest = params_array

            gates = self.set_ctrl_state(
                node, conditional, ctrl_text, bool(controlled_bot)
            )
            gates.append(BoxOnQuWire("CX", conditional=conditional))
            add_connected_gate(node, gates, layer, current_cons)
        # Toffoli gate
        elif isinstance(op.gate, MCX) and op.gate.ctrl_cnt == 2:
            ctrl_text = None
            params_array = TextDrawing.controlled_wires(node, layer)
            controlled_top, controlled_bot, controlled_edge, rest = params_array

            gates = self.set_ctrl_state(
                node, conditional, ctrl_text, bool(controlled_bot)
            )
            gates.append(BoxOnQuWire("Toffoli", conditional=conditional))
            add_connected_gate(node, gates, layer, current_cons)
        # X gate
        elif isinstance(op.gate, XGate):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("X"+condstr, conditional=conditional))
        # BarrierGate gate
        elif isinstance(op.gate, BarrierGate):
            for qubit in node.qargs:
                if qubit in self.qubits:
                    layer.set_qubit(qubit, BarrierBox())
        # CZ gate
        elif isinstance(op.gate, MCZ):
            ctrl_text = None
            params_array = TextDrawing.controlled_wires(node, layer)
            controlled_top, controlled_bot, controlled_edge, rest = params_array

            gates = self.set_ctrl_state(
                node, conditional, ctrl_text, bool(controlled_bot)
            )
            gates.append(BoxOnQuWire("CZ", conditional=conditional))
            add_connected_gate(node, gates, layer, current_cons)
        # CX gate
        elif isinstance(op.gate, MCX):
            ctrl_text = None
            params_array = TextDrawing.controlled_wires(node, layer)
            controlled_top, controlled_bot, controlled_edge, rest = params_array

            gates = self.set_ctrl_state(
                node, conditional, ctrl_text, bool(controlled_bot)
            )
            gates.append(BoxOnQuWire("MCX", conditional=conditional))
            add_connected_gate(node, gates, layer, current_cons)
        elif isinstance(op.gate, P):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("P", conditional=conditional))
        elif isinstance(op.gate, ZGate):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("Z"+condstr, conditional=conditional))
        elif isinstance(op.gate, R):
            label = f"R({op.gate.theta:.2},{op.gate.phi:.2})"
            layer.set_qubit(node.qargs[0], BoxOnQuWire(label, conditional=conditional))
        elif isinstance(op.gate, Rx):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("Rx", conditional=conditional))
        elif isinstance(op.gate, Ry):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("Ry", conditional=conditional))
        elif isinstance(op.gate, Rz):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("Rz", conditional=conditional))
        elif isinstance(op.gate, SGate):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("S", conditional=conditional))
        elif isinstance(op.gate, SdgGate):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("S†", conditional=conditional))
        elif isinstance(op.gate, TGate):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("T", conditional=conditional))
        elif isinstance(op.gate, TdgGate):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("T†", conditional=conditional))
        elif isinstance(op.gate, YGate):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("Y", conditional=conditional))
        # 交互门Swap
        elif isinstance(op.gate, SwapGate):
            gates = [Ex(conditional=conditional) for _ in range(len(node.qargs))]
            add_connected_gate(node, gates, layer, current_cons)
        # 交互控制门CSwap
        elif isinstance(op.gate, CSwapGate):
            ctrl_text = None
            params_array = TextDrawing.controlled_wires(node, layer)
            controlled_top, controlled_bot, controlled_edge, rest = params_array

            gates = self.set_ctrl_state(
                node, conditional, ctrl_text, bool(controlled_bot)
            )
            gates.extend(
                [
                    Ex(conditional=conditional)
                    for _ in range(len(node.qargs) - len(gates))
                ]
            )
            add_connected_gate(node, gates, layer, current_cons)
        # SqrtX gate
        elif isinstance(op.gate, SqrtXGate):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("√X", conditional=conditional))
        # SqrtXdg gate
        elif isinstance(op.gate, SqrtXdgGate):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("√X†", conditional=conditional))
        # CSqrtX gate
        elif isinstance(op.gate, CSqrtXGate):
            ctrl_text = None
            params_array = TextDrawing.controlled_wires(node, layer)
            controlled_top, controlled_bot, controlled_edge, rest = params_array

            gates = self.set_ctrl_state(
                node, conditional, ctrl_text, bool(controlled_bot)
            )
            gates.append(BoxOnQuWire("√X", conditional=conditional))
            add_connected_gate(node, gates, layer, current_cons)
        # CY gate
        elif isinstance(op.gate, CYGate):
            ctrl_text = None
            params_array = TextDrawing.controlled_wires(node, layer)
            controlled_top, controlled_bot, controlled_edge, rest = params_array

            gates = self.set_ctrl_state(
                node, conditional, ctrl_text, bool(controlled_bot)
            )
            gates.append(BoxOnQuWire("CY", conditional=conditional))
            add_connected_gate(node, gates, layer, current_cons)
        # crx gate
        elif isinstance(op.gate, CRx):
            ctrl_text = None
            params_array = TextDrawing.controlled_wires(node, layer)
            controlled_top, controlled_bot, controlled_edge, rest = params_array

            gates = self.set_ctrl_state(
                node, conditional, ctrl_text, bool(controlled_bot)
            )
            gates.append(BoxOnQuWire("Rx(ϴ)", conditional=conditional))
            add_connected_gate(node, gates, layer, current_cons)
        # cry gate
        elif isinstance(op.gate, CRy):
            ctrl_text = None
            params_array = TextDrawing.controlled_wires(node, layer)
            controlled_top, controlled_bot, controlled_edge, rest = params_array

            gates = self.set_ctrl_state(
                node, conditional, ctrl_text, bool(controlled_bot)
            )
            gates.append(BoxOnQuWire("Ry(ϴ)", conditional=conditional))
            add_connected_gate(node, gates, layer, current_cons)
        # crz gate
        elif isinstance(op.gate, CRz):
            ctrl_text = None
            params_array = TextDrawing.controlled_wires(node, layer)
            controlled_top, controlled_bot, controlled_edge, rest = params_array

            gates = self.set_ctrl_state(
                node, conditional, ctrl_text, bool(controlled_bot)
            )
            gates.append(BoxOnQuWire("Rz(λ)", conditional=conditional))
            add_connected_gate(node, gates, layer, current_cons)
        # Rxx gate
        elif isinstance(op.gate, Rxx):
            gate_text = "Rxx(ϴ)"
            layer.set_qu_multibox(node.qargs, gate_text, conditional=conditional)
        # Ryy gate
        elif isinstance(op.gate, Ryy):
            gate_text = "Ryy(ϴ)"
            layer.set_qu_multibox(node.qargs, gate_text, conditional=conditional)
        # Rzz gate
        elif isinstance(op.gate, Rzz):
            gate_text = "Rzz(ϴ)"
            layer.set_qu_multibox(node.qargs, gate_text, conditional=conditional)
        elif isinstance(op.gate, U1):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("U1", conditional=conditional))
        elif isinstance(op.gate, U2):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("U2", conditional=conditional))
        elif isinstance(op.gate, U3):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("U3", conditional=conditional))
        # SqrtSwapGate
        elif isinstance(op.gate, SqrtSwapGate):
            gates = [Ex(conditional=conditional) for _ in range(len(node.qargs))]
            add_connected_gate(node, gates, layer, current_cons)
        # CP gate
        elif isinstance(op.gate, CP):
            ctrl_text = None
            params_array = TextDrawing.controlled_wires(node, layer)
            controlled_top, controlled_bot, controlled_edge, rest = params_array

            gates = self.set_ctrl_state(
                node, conditional, ctrl_text, bool(controlled_bot)
            )
            gates.append(BoxOnQuWire("CP", conditional=conditional))
            add_connected_gate(node, gates, layer, current_cons)
        # X1 gate
        elif isinstance(op.gate, X1Gate):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("X1", conditional=conditional))
        # Y1 gate
        elif isinstance(op.gate, Y1Gate):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("Y1", conditional=conditional))
        # Z1 gate
        elif isinstance(op.gate, Z1Gate):
            layer.set_qubit(node.qargs[0], BoxOnQuWire("Z1", conditional=conditional))
        # CU gate
        elif isinstance(op.gate, CU):
            ctrl_text = None
            params_array = TextDrawing.controlled_wires(node, layer)
            controlled_top, controlled_bot, controlled_edge, rest = params_array

            gates = self.set_ctrl_state(
                node, conditional, ctrl_text, bool(controlled_bot)
            )
            gates.append(BoxOnQuWire("CU", conditional=conditional))
            add_connected_gate(node, gates, layer, current_cons)
        # CU1 gate
        elif isinstance(op.gate, CU1):
            ctrl_text = None
            params_array = TextDrawing.controlled_wires(node, layer)
            controlled_top, controlled_bot, controlled_edge, rest = params_array

            gates = self.set_ctrl_state(
                node, conditional, ctrl_text, bool(controlled_bot)
            )
            gates.append(BoxOnQuWire("CU1", conditional=conditional))
            add_connected_gate(node, gates, layer, current_cons)
        # CU3 gate
        elif isinstance(op.gate, CU3):
            ctrl_text = None
            params_array = TextDrawing.controlled_wires(node, layer)
            controlled_top, controlled_bot, controlled_edge, rest = params_array

            gates = self.set_ctrl_state(
                node, conditional, ctrl_text, bool(controlled_bot)
            )
            gates.append(BoxOnQuWire("CU3", conditional=conditional))
            add_connected_gate(node, gates, layer, current_cons)
        # CR gate
        elif isinstance(op.gate, CR):
            ctrl_text = None
            params_array = TextDrawing.controlled_wires(node, layer)
            controlled_top, controlled_bot, controlled_edge, rest = params_array

            gates = self.set_ctrl_state(
                node, conditional, ctrl_text, bool(controlled_bot)
            )
            gates.append(BoxOnQuWire("CR", conditional=conditional))
            add_connected_gate(node, gates, layer, current_cons)
        # iSwap
        elif isinstance(op.gate, iSwapGate):
            gates = [iEx(conditional=conditional) for _ in range(len(node.qargs))]
            add_connected_gate(node, gates, layer, current_cons)
        # Matrix
        elif isinstance(op.gate, Matrix):
            ctrl_text = None
            params_array = TextDrawing.controlled_wires(node, layer)
            controlled_top, controlled_bot, controlled_edge, rest = params_array

            gates = self.set_ctrl_state(node, conditional, ctrl_text, bool(controlled_bot))

            target_qubits = node.qargs[len(node.op.controls):]
            if len(target_qubits) == 1:
                gates.append(BoxOnQuWire(node.op.gate.name, conditional=conditional))
            else:
                layer.set_qu_multibox(target_qubits, node.op.gate.name, conditional=conditional)

            add_connected_gate(node, gates, layer, current_cons)

            if gates and len(target_qubits) > 1:
                current_cons.append((len(gates), layer.qubit_layer[len(gates)]))
        # Reset Gate
        elif isinstance(op.gate, ResetGate):
            target_qubits = node.qargs[len(node.op.controls):]
            for q in target_qubits:
                layer.set_qubit(q, BoxOnQuWire(node.op.gate.name, conditional=conditional))
        else:
            raise ValueError(
                "Text visualizer does not know how to handle this node: ", op.name
            )

        current_cons.sort(key=lambda tup: tup[0])
        current_cons = [g for q, g in current_cons]

        return layer, current_cons, connection_label

    def build_layers(self):
        """Constructs layers.

        Returns:
           list: List of DrawElements.
        """
        wire_names = self.wire_names()

        if not wire_names:
            return []

        layers = [InputWire.fillup_layer(wire_names)]

        # operator node
        for node_layer in self.nodes:
            # TODO:circuit
            layer = Layer(self.qubits, self.cbits, self.circuit)

            for node in node_layer:
                layer, current_connections, connection_label = self.node_to_gate(
                    node, layer
                )
                layer.connections.append((connection_label, current_connections))

            layer.connect_with("│")
            layers.append(layer.full_layer)

        return layers

    def _get_qubit_labels(self):
        """Qubit name format: q[index], eg: q[1]."""
        qubits = []
        for qubit in self.qubits:
            qubits.append(f"q[{qubit.index}]")
        return qubits

    def _get_cbit_labels(self):
        """Cbit name format: c[index], eg: c[1]."""
        cbits = []
        for cbit in self.cbits:
            cbits.append(f"c[{cbit.index}]")
        return cbits

    # TODO: optimization later
    def wire_names(self, with_initial_state=False):
        """Returns a list of names for each wire.

        Args:
            with_initial_state: If true, adds the initial value to the name.

        Returns:
            list:the list of wire name.
        """

        qubit_labels = self._get_qubit_labels()
        # TODO:doing
        # cbit_labels = self._get_cbit_labels()
        cbit_labels = "c: " + str(len(self.cbits)) + "/"

        if with_initial_state:
            qubit_labels = [f"{qubit}: |0>" for qubit in qubit_labels]
            # cbit_labels = [f"{cbit}: 0 " for cbit in cbit_labels]
        else:
            qubit_labels = [f"{qubit}: " for qubit in qubit_labels]
            # cbit_labels = [f"{cbit}: " for cbit in cbit_labels]

        return qubit_labels + [cbit_labels]

    def should_compress(self, top_line, bot_line):
        """Decides if the top_line and bot_line should be merged,\
            based on `self.vertical_compression`.
        """
        if self.vertical_compression == "high":
            return True

        if self.vertical_compression == "low":
            return False

        # self.vertical_compression == "medium":
        for top, bot in zip(top_line, bot_line):
            if top in ["┴", "╨"] and bot in ["┬", "╥"]:
                return False

        for line in (bot_line, top_line):
            no_spaces = line.replace(" ", "")
            if len(no_spaces) > 0 and all(
                c.isalpha() or c.isnumeric() for c in no_spaces
            ):
                return False

        return True

    @staticmethod
    def normalize_width(layer):
        """When the elements of the layer have different widths, sets the width to the max elements.

        Args:
            layer (list): A list of elements.
        """
        nodes = list(filter(lambda x: x is not None, layer))
        longest = max(node.length for node in nodes)
        for node in nodes:
            node.layer_width = longest

    def draw_wires(self, wires):
        """Creates a list of lines with the text drawing.

        Args:
            wires: [wire1, wire2, ...].

        Returns:
            list:[str1, str2, ...].
        """
        lines = []
        bottom_line = None
        for wire in wires:
            # 1 Build the top of the element
            top_line = ""
            for node in wire:
                top_line += node.top

            if bottom_line is None:
                lines.append(top_line)
            else:
                if self.should_compress(top_line, bottom_line):
                    lines.append(TextDrawing.merge_lines(lines.pop(), top_line))
                else:
                    lines.append(
                        TextDrawing.merge_lines(lines[-1], top_line, icod="bot")
                    )

            # 2 Build the middle of the element
            mid_line = ""
            for node in wire:
                # Build the middle of the element
                mid_line += node.mid
            lines.append(TextDrawing.merge_lines(lines[-1], mid_line, icod="bot"))

            # 3 Build the bottom of the element
            bottom_line = ""
            for node in wire:
                # Build the bottom of the element
                bottom_line += node.bot
            lines.append(TextDrawing.merge_lines(lines[-1], bottom_line, icod="bot"))

        return lines

    @staticmethod
    def merge_lines(top, bot, icod="top"):
        """Merges two lines."""
        ret = ""
        for topc, botc in zip(top, bot):
            if topc == botc:
                ret += topc
            elif topc in "┼╪" and botc == " ":
                ret += "│"
            elif topc == " ":
                ret += botc
            elif topc in "┬╥" and botc in " ║│" and icod == "top":
                ret += topc
            elif topc in "┬" and botc == " " and icod == "bot":
                ret += "│"
            elif topc in "╥" and botc == " " and icod == "bot":
                ret += "║"
            elif topc in "┬│" and botc == "═":
                ret += "╪"
            elif topc in "┬│" and botc == "─":
                ret += "┼"
            elif topc in "└┘║│░" and botc == " " and icod == "top":
                ret += topc
            elif topc in "─═" and botc == " " and icod == "top":
                ret += topc
            elif topc in "─═" and botc == " " and icod == "bot":
                ret += botc
            elif topc in "║╥" and botc in "═":
                ret += "╬"
            elif topc in "║╥" and botc in "─":
                ret += "╫"
            elif topc in "║╫╬" and botc in " ":
                ret += "║"
            elif topc == "└" and botc == "┌" and icod == "top":
                ret += "├"
            elif topc == "┘" and botc == "┐":
                ret += "┤"
            elif botc in "┐┌" and icod == "top":
                ret += "┬"
            elif topc in "┘└" and botc in "─" and icod == "top":
                ret += "┴"
            elif botc == " " and icod == "top":
                ret += topc
            else:
                ret += botc
        return ret

    @staticmethod
    def controlled_wires(node, layer):
        op = node.op
        # number of control qubit
        num_ctrl_qubits = len(op.controls)
        # The front num_ctrl_qubits is the control bit
        ctrl_qubits = node.qargs[:num_ctrl_qubits]
        # The post num_ctrl_qubits is the target bit
        args_qubits = node.qargs[num_ctrl_qubits:]
        ctrl_state = 2**num_ctrl_qubits - 1

        ctrl_state = f"{ctrl_state:b}".rjust(num_ctrl_qubits, "0")[::-1]

        in_box = []
        top_box = []
        bot_box = []

        qubit_index = sorted(i for i, x in enumerate(layer.qubits) if x in args_qubits)

        for ctrl_qubit in zip(ctrl_qubits, ctrl_state):
            if min(qubit_index) > layer.qubits.index(ctrl_qubit[0]):
                top_box.append(ctrl_qubit)
            elif max(qubit_index) < layer.qubits.index(ctrl_qubit[0]):
                bot_box.append(ctrl_qubit)
            else:
                in_box.append(ctrl_qubit)

        return (top_box, bot_box, in_box, args_qubits)

    def set_ctrl_state(self, node, conditional, ctrl_text, bottom):
        op = node.op
        gates = []
        # number of control qubit
        num_ctrl_qubits = len(op.controls)
        # The front num_ctrl_qubits is the control bit
        ctrl_qubits = node.qargs[:num_ctrl_qubits]
        ctrl_state = 2**num_ctrl_qubits - 1

        ctrl_state = f"{ctrl_state:b}".rjust(num_ctrl_qubits, "0")[::-1]

        for i in range(len(ctrl_qubits)):
            if ctrl_state[i] == "1":
                gates.append(
                    Bullet(conditional=conditional, label=ctrl_text, bottom=bottom)
                )
            else:
                gates.append(
                    OpenBullet(conditional=conditional, label=ctrl_text, bottom=bottom)
                )
        return gates


# TODO: have some problems
def get_wire_map(circuit, bits, cregbundle=True):
    """Map the bits and registers to the index.

    Args:
        circuit: The circuit being drawn.
        bits: The list of QuBit or CBit.
        cregbundle: If True bundle classical register. Default: True.

    Returns:
        Dict((QuBit, ClassicalRegister): index )
    """
    prev_reg = None
    wire_index = 0
    wire_map = {}

    for bit in bits:
        # index
        register = get_bit_register(circuit, bit)

        if register is None or not isinstance(bit, CBit) or not cregbundle:
            wire_map[bit] = wire_index
            wire_index += 1

        elif register is not None and cregbundle and register != prev_reg:
            prev_reg = register
            wire_map[register] = wire_index
            wire_index += 1
    #  dict((Qubit, ClassicalRegister): index)
    #  dict((Clbit, ClassicalRegister): index)
    return wire_map


def get_bit_register(circuit, bit):
    """Get the register for a bit.

    Args:
        circuit: The circuit being drawn.
        bit: Qubit or Cbit.

    Returns:
        Index(int)
    """
    return circuit.find_bit(bit)


def get_bit_reg_index(circuit, bit):
    """Get bit index and register information."""
    # index in quantum circuit diagrams
    bit_index = circuit.find_bit(bit)
    # Gets the register and the index in the register
    register = bit.register
    reg_index = bit.index

    return register, bit_index, reg_index
