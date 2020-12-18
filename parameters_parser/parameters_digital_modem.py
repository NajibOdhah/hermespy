from abc import ABC, abstractmethod


class ParametersDigitalModem(ABC):
    """This abstract class implements the parser of the digital modem parameters."""

    @abstractmethod
    def __init__(self) -> None:
        """creates a parsing object, that will manage the digital modem parameters."""
        # Modulation parameters
        self.sampling_rate = 0.

    @abstractmethod
    def read_params(self, file_name: str) -> None:
        """Reads the modem parameters contained in the configuration file 'file_name'."""
        pass

    @abstractmethod
    def _check_params(self) -> None:
        """checks the validity of the parameters."""
        pass
