from abc import ABC, abstractmethod


class Backend(ABC):
    """Basic simulator: All simulators are derived from this class."""

    @abstractmethod
    def send_circuit(self, circuit, final=False):
        """Send the quantum circuit to backend.

        Args:
            circuit: Quantum circuit to send.
            final: True if quantum circuit finish, default False, \
                when final==True The backend program will release the computing resources.
        """
        raise NotImplementedError

    @abstractmethod
    def name(self):
        raise NotImplementedError
