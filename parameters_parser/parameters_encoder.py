
import configparser
from abc import ABC, abstractmethod


class ParametersEncoder(ABC):
    def __init__(self) -> None:
        """creates a parsing object, that will manage the Encoder parameters."""
        self.encoded_bits_n = 1
        self.data_bits_k = 1

    @abstractmethod
    def read_params(self, section: configparser.SectionProxy) -> None:
        """reads the channel parameters contained in the section 'section' of a given configuration file."""
        pass

    @abstractmethod
    def check_params(self) -> None:
        """checks the validity of the parameters."""
        if (self.encoded_bits_n < self.data_bits_k):
            raise ValueError(
                'n must be larger than k, n being the code bits and k the data bits')

        if (self.encoded_bits_n < 1 or self.data_bits_k < 1):
            raise ValueError('n and k must be >= 1')
