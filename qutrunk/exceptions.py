class QuTrunkError(Exception):
    """Base class for errors raised by QuTrunk."""

    def __init__(self, *message):
        """Set the error message.

        Args:
            message: Error message.
        """
        super().__init__(" ".join(message))
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)
