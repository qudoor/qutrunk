"""
The entry for drawing quantum circuit diagram.
"""
from .utils import _get_instructions
from .text import TextDrawing


def circuit_drawer(circuit, output, line_length):
    """Draw the quantum circuit.

    According to the parameters, draw the quantum circuit.

    Args:
        circuit: The quantum circuit to draw.
        output: Select the output method to use for drawing the circuit.
        line_length: The length of the line.

    Returns:
        Drawn string.
    """
    if output == "text":
        return __text_circuit_drawer(circuit=circuit, line_length=line_length)


def __text_circuit_drawer(
    circuit, plot_barriers=True, vertical_compression="high", line_length=None
):
    """Draw quantum circuit by ascii art.

    Args:
        circuit: Input circuit.
        plot_barriers: Whether to export barriers.
        vertical_compression: Vertical compression with text (default: Medium).
        line_length: The maximum length of a line when drawing.

    Returns:
        Return the draw result.

    """
    qubits, cbits, nodes = _get_instructions(circuit)
    text_drawing = TextDrawing(qubits=qubits, cbits=cbits, nodes=nodes, circuit=circuit)
    text_drawing.plot_barriers = plot_barriers
    text_drawing.line_length = line_length
    text_drawing.vertical_compression = vertical_compression

    return text_drawing
