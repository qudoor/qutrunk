"""Exceptions for errors raised by local backend."""

from qutrunk.exceptions import QuTrunkError


class LocalBackendError(QuTrunkError):
    """Base class for errors raised by Local backend."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)
