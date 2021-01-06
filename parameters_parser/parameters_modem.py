import configparser
import ast
from abc import ABC, abstractmethod
from typing import List, Any

import numpy as np


class ParametersModem(ABC):
    """This abstract class implements the parser of the transceiver parameters.

    Attributes:
        technology_val list(str): valid technology values
        technology (ParametersWaveformGenerator): object containing all technology-specific parameters
        position (list(float)): 3D-location of transceiver (in meters)
        velocity (list(float)): 3D- velocity of transceiver (in m/s)
        number_of_antennas (int): number of tx/rx antennas
        carrier_frequency (float): transceiver carrier frequency (in Hz)
    """

    technology_val = ["PSK_QAM", "CHIRP_FSK", "OFDM"]

    def __init__(self) -> None:
        """creates a parsing object, that will manage the transceiver parameters."""
        self.position: List[float] = []
        self.velocity = np.array([])
        self.number_of_antennas = 1
        self.carrier_frequency = 0.
        self.tx_power = 0.
        self.technology: Any = None

    @abstractmethod
    def read_params(self, section: configparser.SectionProxy) -> None:
        """reads the channel parameters contained in the section 'section' of a given configuration file."""
        self.position = ast.literal_eval(section.get("position"))
        self.velocity = ast.literal_eval(section.get("velocity"))
        self.number_of_antennas = section.getint("number_of_antennas")

        tx_power_db = section.getfloat("tx_power_db", fallback=0.)
        if tx_power_db == 0:
            self.tx_power = 0.
        else:
            self.tx_power = 10 ** (tx_power_db / 10.)

    @abstractmethod
    def check_params(self) -> None:
        """checks the validity of the parameters."""
        if (not isinstance(self.position, list) or not len(self.position) == 3 or
                not all(isinstance(x, (int, float)) for x in self.position)):
            raise ValueError(
                'position (' +
                ' '.join(
                    str(e) for e in self.position) +
                ' must be a 3-D number vector')

        if (not isinstance(self.velocity, list) or not len(self.velocity) == 3 or
                not all(isinstance(x, (int, float)) for x in self.velocity)):
            raise ValueError(
                'velocity (' +
                ' '.join(
                    str(e) for e in self.velocity) +
                ' must be a 3-D number vector')

        self.velocity = np.asarray(self.velocity)

        if not isinstance(self.number_of_antennas,
                          int) or self.number_of_antennas < 1:
            raise ValueError('number_of_antennas (' +
                             str(self.number_of_antennas) +
                             'must be an integer > 0')
