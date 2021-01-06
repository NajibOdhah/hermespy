import configparser
from enum import Enum
from collections import namedtuple
from typing import List

import numpy as np

from parameters_parser.parameters_waveform_generator import ParametersWaveformGenerator

FrameElementDef = namedtuple('FrameElementDef', 'type number_of_samples')


class ParametersOfdm(ParametersWaveformGenerator):
    """This class implements the parameters parser of an OFDM modem

    Attributes:
        subcarrier_spacing(float): frequency spacing between adjacent subcarriers
        fft_size(int): number of elements in FFT
        number_occupied_subcarriers(int): number of subcarriers that are occupied with data/pilot symbols. Subcarriers
            are allocated in the middle of the spectrum, except the DC subcarrier
        guard_interval_ratio(numpy.ndarray): size of guard interval relative to the OFDM symbol duration
        precoding(str): type of precoding. Allowed values are 'NONE' or 'DFT'. If 'DFT', then DFT-spread (SC-FDMA)
            modulation is employed.
        precoding_val(list of str, class level): list of the valid precoding types
        modulation_order: dimension of PSK/QAM modulation constellation. Allowed values are 2, 4, 16, 64, 256.
        modulation_order_val(list of int, class level): list of valid modulation orders
        oversampling_factor(str): signal can be upsampled relative to the critical sampling rate. The sampling rate will
            be 'subcarrier_spacing' * 'fft_size' * 'oversampling_factor'

        link_direction(str): direction for which transmission will be simulated. Supported values are 'UPLINK' and
            'DOWNLINK'. 'UL' and 'DL' are accepted, but will be internally converted to 'UPLINK' and 'DOWNLINK'.
        link_direction(list of str, class level): list of valid direction values
        non_link_direction(str): direction opposite to the one that will be simulated
        frame_guard_interval(float): length of guard interval inside a frame
        frame_structure(list of tuples): description of the frame structure. It is a list of tuples containing the
            blocks that form a transmission frame. Each tuple contains two elements, the first should be a string
            describing the type of each block, and the second should be an int with the number of FFT samples of each
            block. The possible block types are: 'REFERENCE', 'DOWNLINK', 'UPLINK', CYCLIC_PREFIX', 'ZERO_PREFIX',
            'GUARD_INTERVAL'.

        pilot_subcarriers(list of numpy.ndarray): list of arrays containing the positions of the pilot symbols inside
            the frame in the frequency domain. Each array should have the dimension T x P, with T the number of transmit
            antennas and P the number of pilots. Each element in the list corresponds to a data OFDM symbol. If the list
            size is less than the number of data OFDM symbols in the frame, than the pilot sequence is repeated. If
            empty, then no pilots are considered.
        pilot_symbols(list of numpy.ndarray): list of arrays containing the values of the pilot symbols. This list
            should have the same size as 'pilot_subcarriers'
        reference_symbols(list of numpy.ndarray): list of arrays containing the values of the reference symbols in the
            frequency domain. Each array should have the dimension T x N, with T the number of transmit antennas and N
            the FFT size.
        bits_in_ofdm_symbol:
        bits_in_frame:
    """

    modulation_order_val = [2, 4, 16, 64, 256]
    direction_val = ['DOWNLINK', 'DL', 'UPLINK', 'UL']
    precoding_val = ['NONE', 'DFT']
    channel_estimation_val = [
        'IDEAL',
        'IDEAL_PREAMBLE',
        'IDEAL_MIDAMBLE',
        'IDEAL_POSTAMBLE']
    equalization_val = ['ZF', 'MMSE']
    element_types = Enum('element_types',
                         ['DOWNLINK', 'UPLINK', 'GUARD_INTERVAL', 'CYCLIC_PREFIX', 'ZERO_PREFIX', 'REFERENCE'])

    def __init__(self) -> None:
        """creates a parsing object, that will manage the transceiver parameters.
        """
        super().__init__()

        # Modulation parameters
        self.subcarrier_spacing = 0.
        self.fft_size = 1
        self.number_occupied_subcarriers = 1
        self.guard_interval_ratio = np.array([])
        self.precoding = ""
        self.modulation_order = 2
        self.oversampling_factor = 1
        self.bits_in_ofdm_symbol = 1
        self.bits_in_frame = 0

        # receiver parameters
        self.channel_estimation = ""
        self.equalization = ""

        # Frame parameters
        self.link_direction = ParametersOfdm.element_types.DOWNLINK
        self.non_link_direction = ParametersOfdm.element_types.UPLINK
        self.frame_guard_interval = 0.
        self.frame_structure: List[FrameElementDef] = []

        # technology-specific pilot and reference signals can be defined here
        # to be used by technology-specific derived classes
        self.pilot_subcarriers: List[np.ndarray] = []
        # content of pilot subcarriers
        self.pilot_symbols: List[np.ndarray] = []
        # reference signal (in frequency domain)
        self.reference_symbols: List[np.ndarray] = []
        self._link_direction_str = ""

    def read_params(self, file_name: str) -> None:
        """reads the modem parameters

        Args:
            file_name(str): name/path of configuration file containing the parameters
        """

        super().read_params(file_name)

        config = configparser.ConfigParser()
        config.read(file_name)

        cfg = config['Modulation']

        self.subcarrier_spacing = cfg.getfloat('subcarrier_spacing')
        self.fft_size = cfg.getint('fft_size')
        self.oversampling_factor = cfg.getint("oversampling_factor")

        self.sampling_rate = self.subcarrier_spacing * \
            self.fft_size * self.oversampling_factor

        self.number_occupied_subcarriers = cfg.getint(
            'number_occupied_subcarriers')

        self.guard_interval_ratio = cfg.get('guard_interval_ratio')
        self.guard_interval_ratio = np.fromstring(
            self.guard_interval_ratio, sep=',')

        self.modulation_order = cfg.getint("modulation_order")
        self.bits_in_ofdm_symbol = self.number_occupied_subcarriers * \
            int(np.log2(self.modulation_order))

        self.precoding = cfg.get('precoding')

        cfg = config['Receiver']

        self.channel_estimation = cfg.get('channel_estimation')
        self.equalization = cfg.get('equalization')

        cfg = config['Frame']

        self._link_direction_str = cfg.get('link_direction')

        self.frame_guard_interval = cfg.getfloat("frame_guard_interval")
        self._frame_structure_str = cfg.get("frame_structure")

        self._check_params()

    def _check_params(self) -> None:
        """checks the validity of the parameters and calculates derived parameters

        Raises:
            ValueError: if there is an invalid value in the parameters
        """

        top_header = 'ERROR reading OFDM modem parameters'

        #######################
        # check modulation parameters
        msg_header = top_header + ', Section "Modulation", '
        if self.sampling_rate <= 0:
            raise ValueError(
                msg_header +
                f'sampling rate ({self.sampling_rate}) must be positive')

        if self.number_occupied_subcarriers > self.fft_size:
            raise ValueError(msg_header + (f'number_occupied_subcarriers({self.number_occupied_subcarriers})'
                                           f' cannot be larger than fft_size({self.fft_size})'))

        if self.modulation_order not in ParametersOfdm.modulation_order_val:
            raise ValueError(
                msg_header +
                f'modulation order ({self.modulation_order}) not supported')

        self.precoding = self.precoding.upper()
        if self.precoding not in ParametersOfdm.precoding_val:
            raise ValueError(
                msg_header +
                f'precoding ({self.precoding}) not supported')

        #############################
        # check receiver parameters
        msg_header = top_header + ', Section "Receiver", '
        self.channel_estimation = self.channel_estimation.upper()
        if self.channel_estimation not in ParametersOfdm.channel_estimation_val:
            raise ValueError(
                msg_header +
                f'channel_estimation ({self.channel_estimation}) not supported')

        if self.equalization not in ParametersOfdm.equalization_val:
            raise ValueError(
                msg_header +
                f'equalization ({self.equalization}) not supported')
        #############################
        # check frame parameters
        msg_header = top_header + ', Section "Frame", '

        self._link_direction_str = self._link_direction_str.upper()
        if self._link_direction_str not in ParametersOfdm.direction_val:
            raise ValueError(
                msg_header +
                f'link_direction ({self._link_direction_str}) not supported')

        if self._link_direction_str == 'DL' or self._link_direction_str == 'DOWNLINK':
            self.link_direction = ParametersOfdm.element_types.DOWNLINK
            self.non_link_direction = ParametersOfdm.element_types.UPLINK

        elif self._link_direction_str == 'UL' or self._link_direction_str == 'UPLINK':
            self.link_direction = ParametersOfdm.element_types.UPLINK
            self.non_link_direction = ParametersOfdm.element_types.DOWNLINK

        #######################################################################
        # read the frame structure, which will be a list of tuples describing the frame, which consists of a sequence of
        # elements
        # each tuple contain two fields, the first one describes the type of the element (cyclic prefix, downlink data,
        # etc), and the second one is the number of samples (after IFFT,
        # without upsampling) for this element
        self._frame_structure_str = self._frame_structure_str.replace(' ', '')
        self._frame_structure_str = self._frame_structure_str.upper()
        frame_structure = self._frame_structure_str.split(',')

        for subframe_str in frame_structure:
            number_of_repetitions = 0
            while subframe_str[0].isdigit():
                number_of_repetitions = 10 * \
                    number_of_repetitions + int(subframe_str[0])
                subframe_str = subframe_str[1:]

            if number_of_repetitions > 0:
                if subframe_str[0] != '(' or subframe_str[-1] != ')':
                    raise ValueError(
                        msg_header + 'error in frame structure definition, parenthesis required')
                else:
                    subframe_str = subframe_str[1:-1]
            else:
                number_of_repetitions = 1

            subframe = []
            while len(subframe_str):
                element_str = subframe_str[0]
                subframe_str = subframe_str[1:]

                if element_str == 'R':
                    element = FrameElementDef(
                        ParametersOfdm.element_types.REFERENCE, self.fft_size)
                elif element_str == 'D':
                    element = FrameElementDef(
                        ParametersOfdm.element_types.DOWNLINK, self.fft_size)
                elif element_str == 'U':
                    element = FrameElementDef(
                        ParametersOfdm.element_types.UPLINK, self.fft_size)
                elif element_str == 'C' or element_str == 'Z':
                    prefix_index = 0
                    while subframe_str[0].isdigit():
                        prefix_index = 10 * prefix_index + int(subframe_str[0])
                        subframe_str = subframe_str[1:]

                    number_of_samples = int(
                        np.around(
                            self.guard_interval_ratio[prefix_index] *
                            self.fft_size))

                    if element_str == 'C':
                        element = FrameElementDef(
                            ParametersOfdm.element_types.CYCLIC_PREFIX, number_of_samples)
                    elif element_str == 'Z':
                        element = FrameElementDef(
                            ParametersOfdm.element_types.ZERO_PREFIX, number_of_samples)
                elif element_str == 'G':
                    number_of_samples = int(np.around(self.frame_guard_interval * self.subcarrier_spacing *
                                                      self.fft_size))
                    element = FrameElementDef(
                        ParametersOfdm.element_types.GUARD_INTERVAL, number_of_samples)
                else:
                    raise ValueError(msg_header + "error in frame structure definition, element type '" + element_str
                                     + "' not supported")

                subframe.append(element)

            for idx in range(number_of_repetitions):
                self.frame_structure += subframe

        for frame_element in self.frame_structure:
            if frame_element.type == self.link_direction:
                self.bits_in_frame += self.bits_in_ofdm_symbol

        #######################################################################
