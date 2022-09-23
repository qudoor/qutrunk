"""Exception for errors raised while parsing OPENQASM."""
from qutrunk.exceptions import QuTrunkError


class QasmError(QuTrunkError):
    """Base class for errors raised while parsing OPENQASM."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
