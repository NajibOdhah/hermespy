import numpy as np
from numpy import random as rnd

from parameters_parser.parameters_modem import ParametersModem


class RfChain(object):
    """Implements an RF chain model.

    It is currently only a dummy class, no impairments are implemented.
    """

    def __init__(self, param: ParametersModem,
                 random_number_gen: rnd.RandomState) -> None:
        pass

    def send(self, input_signal: np.ndarray) -> np.ndarray:
        """Returns the distorted version of signal in "input_signal".

        According to transmission impairments.
        """
        return input_signal

    def receive(self, input_signal: np.ndarray) -> np.ndarray:
        """Returns the distorted version of signal in "input_signal".

        According to reception impairments.
        """
        return input_signal
