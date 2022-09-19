class DrawElement:
    """Draw Element.
    """

    def __init__(self, label=None):
        self._width = None
        self.label = self.mid_content = label
        self.top_format = self.mid_format = self.bot_format = "%s"
        self.top_connect = self.bot_connect = " "
        self.top_pad = self._mid_padding = self.bot_pad = " "
        self.mid_bck = self.top_bck = self.bot_bck = " "
        self.bot_connector = {}
        self.top_connector = {}
        self.right_fill = self.left_fill = self.layer_width = 0
        self.wire_label = ""

    @property
    def top(self):
        """Constructs the top line of the element."""
        if (self.width % 2) == 0 and len(self.top_format) % 2 == 1 and len(self.top_connect) == 1:
            ret = self.top_format % (self.top_pad + self.top_connect).center(
                self.width, self.top_pad
            )
        else:
            ret = self.top_format % self.top_connect.center(self.width, self.top_pad)
        if self.right_fill:
            ret = ret.ljust(self.right_fill, self.top_pad)
        if self.left_fill:
            ret = ret.rjust(self.left_fill, self.top_pad)
        ret = ret.center(self.layer_width, self.top_bck)
        return ret

    @property
    def mid(self):
        """Constructs the middle line of the element."""
        ret = self.mid_format % self.mid_content.center(self.width, self._mid_padding)
        if self.right_fill:
            ret = ret.ljust(self.right_fill, self._mid_padding)
        if self.left_fill:
            ret = ret.rjust(self.left_fill, self._mid_padding)
        ret = ret.center(self.layer_width, self.mid_bck)
        return ret

    @property
    def bot(self):
        """Constructs the bottom line of the element."""
        if (self.width % 2) == 0 and len(self.top_format) % 2 == 1:
            ret = self.bot_format % (self.bot_pad + self.bot_connect).center(
                self.width, self.bot_pad
            )
        else:
            ret = self.bot_format % self.bot_connect.center(self.width, self.bot_pad)
        if self.right_fill:
            ret = ret.ljust(self.right_fill, self.bot_pad)
        if self.left_fill:
            ret = ret.rjust(self.left_fill, self.bot_pad)
        ret = ret.center(self.layer_width, self.bot_bck)
        return ret

    @property
    def length(self):
        """Returns the length of the element, including the box around."""
        return max(len(self.top), len(self.mid), len(self.bot))

    @property
    def width(self):
        """Returns the width of the label, including padding."""
        if self._width:
            return self._width
        return len(self.mid_content)

    @width.setter
    def width(self, value):
        self._width = value

    def connect(self, wire_char, where, label=None):
        """Connects boxes and elements using wire_char and setting proper connectors.

        Args:
            wire_char (char): For example '║' or '│'.
            where (list["top", "bot"]): Where the connector should be set.
            label (string): Some connectors have a label (see cu1, for example).
        """

        if "top" in where and self.top_connector:
            self.top_connect = self.top_connector[wire_char]

        if "bot" in where and self.bot_connector:
            self.bot_connect = self.bot_connector[wire_char]

        if label:
            self.top_format = self.top_format[:-1] + (label if label else "")


class BoxOnClWire(DrawElement):
    """Draws a box on the classical wire.

    Args:
        label: Used as mid_content.
        top_connect: top_connect.
        bot_connect: bot_connect.

    Example:
    ::

        top: ┌───┐   ┌───┐
        mid: ╡ A ╞ ══╡ A ╞══
        bot: └───┘   └───┘
    """
    def __init__(self, label="", top_connect="─", bot_connect="─"):
        super().__init__(label)
        self.top_format = "┌─%s─┐"
        self.mid_format = "╡ %s ╞"
        self.bot_format = "└─%s─┘"
        self.top_pad = self.bot_pad = "─"
        self.mid_bck = "═"
        self.top_connect = top_connect
        self.bot_connect = bot_connect
        self.mid_content = label


class BoxOnQuWire(DrawElement):
    """Draws a box on the quantum wire.

    Example:
    ::

        top: ┌───┐   ┌───┐
        mid: ┤ A ├ ──┤ A ├──
        bot: └───┘   └───┘
    """

    def __init__(self, label="", top_connect="─", conditional=False):
        super().__init__(label)
        self.top_format = "┌─%s─┐"
        self.mid_format = "┤ %s ├"
        self.bot_format = "└─%s─┘"
        self.top_pad = self.bot_pad = self.mid_bck = "─"
        self.top_connect = top_connect
        self.bot_connect = "╥" if conditional else "─"
        self.mid_content = label
        self.top_connector = {"│": "┴"}
        self.bot_connector = {"│": "┬"}


class MeasureTo(DrawElement):
    """The element on the classic wire to which the measure is performed.

    Example:
    ::

        top:  ║     ║
        mid: ═╩═ ═══╩═══
        bot:
    """

    def __init__(self, label=""):
        super().__init__()
        self.top_connect = " ║ "
        self.mid_content = "═╩═"
        self.bot_connect = label
        self.mid_bck = "═"


class MeasureFrom(BoxOnQuWire):
    """The element on the quantum wire in which the measure is performed.

    Example:
    ::

        top: ┌─┐    ┌─┐
        mid: ┤M├ ───┤M├───
        bot: └╥┘    └╥┘
    """

    def __init__(self):
        super().__init__()
        self.top_format = self.mid_format = self.bot_format = "%s"
        self.top_connect = "┌─┐"
        self.mid_content = "┤M├"
        self.bot_connect = "└╥┘"

        self.top_pad = self.bot_pad = " "
        self._mid_padding = "─"


class MultiBox(DrawElement):
    """Elements that are drawn over multiple wires."""

    def center_label(self, input_length, order):
        """In multi-bit elements, the label is centered vertically."""
        if input_length == order == 0:
            self.top_connect = self.label
            return

        location_in_the_box = "*".center(input_length * 2 - 1).index("*") + 1
        top_limit = order * 2 + 2
        bot_limit = top_limit + 2

        if top_limit <= location_in_the_box < bot_limit:
            if location_in_the_box == top_limit:
                self.top_connect = self.label
            elif location_in_the_box == top_limit + 1:
                self.mid_content = self.label
            else:
                self.bot_connect = self.label

    @property
    def width(self):
        """Returns the width of the label, including padding."""
        if self._width:
            return self._width
        return len(self.label)


class BoxOnQuWireTop(MultiBox, BoxOnQuWire):
    """Draws the top part of a box that affects more than one quantum wire.
    """

    def __init__(self, label="", top_connect=None, wire_label=""):
        super().__init__(label)
        self.wire_label = wire_label
        self.bot_connect = self.bot_pad = " "
        self.mid_content = ""
        self.left_fill = len(self.wire_label)
        self.top_format = "┌─" + "s".center(self.left_fill + 1, "─") + "─┐"
        self.top_format = self.top_format.replace("s", "%s")
        self.mid_format = f"┤{self.wire_label} %s ├"
        self.bot_format = f"│{self.bot_pad * self.left_fill} %s │"
        self.top_connect = top_connect if top_connect else "─"


class BoxOnWireMid(MultiBox):
    """A generic middle box.
    """

    def __init__(self, label, input_length, order, wire_label=""):
        super().__init__(label)
        self.top_pad = self.bot_pad = self.top_connect = self.bot_connect = " "
        self.wire_label = wire_label
        self.left_fill = len(self.wire_label)
        self.top_format = f"│{self.top_pad * self.left_fill} %s │"
        self.bot_format = f"│{self.bot_pad * self.left_fill} %s │"
        self.top_connect = self.bot_connect = self.mid_content = ""
        self.center_label(input_length, order)


class BoxOnQuWireMid(BoxOnWireMid, BoxOnQuWire):
    """Draws the middle part of a box that affects more than one quantum wire.
    """

    def __init__(self, label, input_length, order, wire_label="", control_label=None):
        super().__init__(label, input_length, order, wire_label=wire_label)
        if control_label:
            self.mid_format = f"{control_label}{self.wire_label} %s ├"
        else:
            self.mid_format = f"┤{self.wire_label} %s ├"


class BoxOnQuWireBot(MultiBox, BoxOnQuWire):
    """Draws the bottom part of a box that affects more than one quantum wire.
    """

    def __init__(self, label, input_length, bot_connect=None, wire_label="", conditional=False):
        super().__init__(label)
        self.wire_label = wire_label
        self.top_pad = " "
        self.left_fill = len(self.wire_label)
        self.top_format = f"│{self.top_pad * self.left_fill} %s │"
        self.mid_format = f"┤{self.wire_label} %s ├"
        self.bot_format = "└─" + "s".center(self.left_fill + 1, "─") + "─┘"
        self.bot_format = self.bot_format.replace("s", "%s")
        bot_connect = bot_connect if bot_connect else "─"
        self.bot_connect = "╥" if conditional else bot_connect

        self.mid_content = self.top_connect = ""
        if input_length <= 2:
            self.top_connect = label


class BoxOnClWireTop(MultiBox, BoxOnClWire):
    """Draws the top part of a conditional box that affects more than one classical wire.
    """

    def __init__(self, label="", top_connect=None, wire_label=""):
        super().__init__(label)
        self.wire_label = wire_label
        self.mid_content = ""
        self.bot_format = "│ %s │"
        self.top_connect = top_connect if top_connect else "─"
        self.bot_connect = self.bot_pad = " "


class BoxOnClWireMid(BoxOnWireMid, BoxOnClWire):
    """Draws the middle part of a conditional box that affects more than one classical wire.
    """

    def __init__(self, label, input_length, order, wire_label="", **_):
        super().__init__(label, input_length, order, wire_label=wire_label)
        self.mid_format = f"╡{self.wire_label} %s ╞"


class BoxOnClWireBot(MultiBox, BoxOnClWire):
    """Draws the bottom part of a conditional box that affects more than one classical wire.
    """

    def __init__(self, label, input_length, bot_connect="─", wire_label="", **_):
        super().__init__(label)
        self.wire_label = wire_label
        self.left_fill = len(self.wire_label)
        self.top_pad = " "
        self.bot_pad = "─"
        self.top_format = f"│{self.top_pad * self.left_fill} %s │"
        self.mid_format = f"╡{self.wire_label} %s ╞"
        self.bot_format = "└─" + "s".center(self.left_fill + 1, "─") + "─┘"
        self.bot_format = self.bot_format.replace("s", "%s")
        bot_connect = bot_connect if bot_connect else "─"
        self.bot_connect = bot_connect

        self.mid_content = self.top_connect = ""
        if input_length <= 2:
            self.top_connect = label


class DirectOnQuWire(DrawElement):
    """Element to the wire (without the box).
    """

    def __init__(self, label=""):
        super().__init__(label)
        self.top_format = " %s "
        self.mid_format = "─%s─"
        self.bot_format = " %s "
        self._mid_padding = self.mid_bck = "─"
        self.top_connector = {"│": "│"}
        self.bot_connector = {"│": "│"}


class Barrier(DirectOnQuWire):
    """Draws a barrier.

    Example:

        top:  ░     ░
        mid: ─░─ ───░───
        bot:  ░     ░
    """

    def __init__(self, label=""):
        super().__init__("░")
        self.top_connect = "░"
        self.bot_connect = "░"
        self.top_connector = {}
        self.bot_connector = {}


class Ex(DirectOnQuWire):
    """Draws an X (usually with a connector).

    Example:
        top:
        mid: ─X─ ───X───
        bot:  │     │
    """

    def __init__(self, bot_connect=" ", top_connect=" ", conditional=False):
        super().__init__("X")
        self.bot_connect = "║" if conditional else bot_connect
        self.top_connect = top_connect


class iEx(DirectOnQuWire):
    """Draws an ⨂ (usually with a connector).

    Example:

        top:
        mid: ─⨂─ ───⨂───
        bot:  │      │
    """

    def __init__(self, bot_connect="  ", top_connect=" ", conditional=False):
        super().__init__("⨂")
        self.bot_connect = "║" if conditional else bot_connect
        self.top_connect = top_connect

class ResetDisplay(DirectOnQuWire):
    """Draws a reset gate.
    """

    def __init__(self, conditional=False):
        super().__init__("|0>")
        if conditional:
            self.bot_connect = "║"


class Bullet(DirectOnQuWire):
    """Draws a bullet (usually with a connector).

    Example:
        top:
        mid: ─■─  ───■───
        bot:  │      │
    """

    def __init__(self, top_connect="", bot_connect="", conditional=False, label=None, bottom=False):
        super().__init__("■")
        self.conditional = conditional
        self.top_connect = top_connect
        self.bot_connect = "║" if conditional else bot_connect
        if label and bottom:
            self.bot_connect = label
        elif label:
            self.top_connect = label
        self.mid_bck = "─"


class OpenBullet(DirectOnQuWire):
    """Draws an open bullet (usually with a connector).

    Example:
        top:
        mid: ─o─  ───o───
        bot:  │      │
    """

    def __init__(self, top_connect="", bot_connect="", conditional=False, label=None, bottom=False):
        super().__init__("o")
        self.conditional = conditional
        self.top_connect = top_connect
        self.bot_connect = "║" if conditional else bot_connect
        if label and bottom:
            self.bot_connect = label
        elif label:
            self.top_connect = label
        self.mid_bck = "─"


class DirectOnClWire(DrawElement):
    """Element to the classical wire (without the box).
    """

    def __init__(self, label=""):
        super().__init__(label)
        self.top_format = " %s "
        self.mid_format = "═%s═"
        self.bot_format = " %s "
        self._mid_padding = self.mid_bck = "═"
        self.top_connector = {"│": "│"}
        self.bot_connector = {"│": "│"}


class ClBullet(DirectOnClWire):
    """Draws a bullet on classical wire (usually with a connector).

    Example:
        top:
        mid: ═■═  ═══■═══
        bot:  │      │
    """

    def __init__(self, top_connect="", bot_connect="", conditional=False, label=None, bottom=False):
        super().__init__("■")
        self.top_connect = top_connect
        self.bot_connect = "║" if conditional else bot_connect
        if label and bottom:
            self.bot_connect = label
        elif label:
            self.top_connect = label
        self.mid_bck = "═"


class ClOpenBullet(DirectOnClWire):
    """Draws an open bullet on classical wire (usually with a connector).

    Example:
        top:
        mid: ═o═  ═══o═══
        bot:  │      │
    """

    def __init__(self, top_connect="", bot_connect="", conditional=False, label=None, bottom=False):
        super().__init__("o")
        self.top_connect = top_connect
        self.bot_connect = "║" if conditional else bot_connect
        if label and bottom:
            self.bot_connect = label
        elif label:
            self.top_connect = label
        self.mid_bck = "═"


class EmptyWire(DrawElement):
    """This element is just the wire, with no operations.
    """

    def __init__(self, wire):
        super().__init__(wire)
        self._mid_padding = self.mid_bck = wire

    @staticmethod
    def fillup_layer(layer, first_clbit):
        """Given a layer, replace the Nones in it with EmptyWire elements.
        """
        for nones in [i for i, x in enumerate(layer) if x is None]:
            layer[nones] = EmptyWire("═") if nones >= first_clbit else EmptyWire("─")
        return layer


class BreakWire(DrawElement):
    """This element is used to break the drawing in several pages.
    """

    def __init__(self, arrow_char):
        super().__init__()
        self.top_format = self.mid_format = self.bot_format = "%s"
        self.top_connect = arrow_char
        self.mid_content = arrow_char
        self.bot_connect = arrow_char

    @staticmethod
    def fillup_layer(layer_length, arrow_char):
        """Creates a layer with BreakWire elements.
        """
        breakwire_layer = []
        for _ in range(layer_length):
            breakwire_layer.append(BreakWire(arrow_char))
        return breakwire_layer


class InputWire(DrawElement):
    """This element is the label and the initial value of a wire.
    """

    def __init__(self, label):
        super().__init__(label)

    @staticmethod
    def fillup_layer(names):
        """Creates a layer with InputWire elements.
        """
        longest = max(len(name) for name in names)
        inputs_wires = []
        for name in names:
            inputs_wires.append(InputWire(name.rjust(longest)))
        return inputs_wires
