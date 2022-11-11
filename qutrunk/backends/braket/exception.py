"""QuTrunk-Braket exception."""

from qutrunk.exceptions import QuTrunkError


class QuTrunkBraketException(QuTrunkError):
    """QuTrunk-Braket exception."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
