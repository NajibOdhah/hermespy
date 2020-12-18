from typing import List, Tuple

import numpy as np

from modem.digital_modem import DigitalModem
from parameters_parser.parameters_psk_qam import ParametersPskQam
from modem.tools.shaping_filter import ShapingFilter
from modem.tools.psk_qam_mapping import PskQamMapping


class DigitalModemPskQam(DigitalModem):
    """This method provides a class for a generic PSK/QAM modem.

    The modem has the following characteristics:
    - root-raised cosine filter with arbitrary roll-off factor
    - arbitrary constellation, as defined in modem.tools.psk_qam_mapping:PskQamMapping

    This implementation has currently the following limitations:
    - SISO only
    - hard output only (no LLR)
    - no reference signal
    - ideal channel estimation
    - no equalization (only amplitude and phase of first propagation path is compensated)
    """
    # BitsSource is no longer required

    def __init__(self, param: ParametersPskQam) -> None:
        """Creates a modem object

        Args:
            param(ParametersPskQam): object containing all the relevant parameters
            source(BitsSource): bits source for transmitter
        """
        super().__init__(param)
        self.param = param
        self._set_frame_derived_parameters()
        self._set_filters()
        self._set_sampling_indices()

    def create_frame(self, timestamp: int,
                     data_bits: np.array) -> Tuple[np.ndarray, int, int]:
        frame = np.zeros(self._samples_in_frame, dtype=complex)
        frame[self._symbol_idx[:self.param.number_preamble_symbols]] = 1
        start_index_data = self.param.number_preamble_symbols
        end_index_data = self.param.number_preamble_symbols + self.param.number_data_symbols
        frame[self._symbol_idx[start_index_data: end_index_data]
              ] = self._mapping.get_symbols(data_bits)
        frame[self._symbol_idx[end_index_data:]] = 1

        output_signal = self._filter_tx.filter(frame)

        initial_sample_num = timestamp - self._filter_tx.delay_in_samples
        timestamp += self._samples_in_frame

        return output_signal, timestamp, initial_sample_num

    def receive_frame(self,
                      rx_signal: np.ndarray,
                      timestamp_in_samples: int,
                      snr_linear_esn0: float) -> Tuple[List[np.ndarray],
                                                       np.ndarray]:
        useful_signal_length = self._samples_in_frame + self._filter_rx.delay_in_samples

        if rx_signal.shape[1] < useful_signal_length:
            bits = None
            rx_signal = np.array([])
        else:
            frame_signal = rx_signal[0, :useful_signal_length].ravel()
            frame_signal = self._filter_rx.filter(frame_signal)

            symbol_idx = self._data_symbol_idx + \
                self._filter_rx.delay_in_samples + self._filter_tx.delay_in_samples

            # consider ideal channel knowledge (compensate phase/amplitude of
            # only first path)
            timestamps = (timestamp_in_samples + symbol_idx) / \
                self.param.sampling_rate
            channel = self._channel.get_impulse_response(timestamps)
            channel = channel[:, :, :, 0].ravel()

            bits = self._mapping.detect_bits(
                frame_signal[symbol_idx] / channel)

            rx_signal = rx_signal[:, self._samples_in_frame:]

        return list([bits]), rx_signal

    def _set_frame_derived_parameters(self) -> None:
        """ Derives local frame-specific parameter based on parameter class
        """
        # derived parameters
        if self.param.number_preamble_symbols > 0 or self.param.number_postamble_symbols > 0:
            self._samples_in_pilot = int(
                np.round(
                    self.param.sampling_rate /
                    self.param.pilot_symbol_rate))
        else:
            self._samples_in_pilot = 0

        self._samples_in_guard = int(
            np.round(
                self.param.guard_interval *
                self.param.sampling_rate))

        self._samples_in_frame = int(
            (self.param.number_preamble_symbols +
             self.param.number_postamble_symbols) *
            self._samples_in_pilot +
            self._samples_in_guard +
            self.param.oversampling_factor *
            self.param.number_data_symbols)
        self._mapping = PskQamMapping(self.param.modulation_order)

    def _set_filters(self) -> None:
        """ Initializes transmit and reception filters based on parameter class
        """
        self._filter_tx = ShapingFilter(
            self.param.filter_type,
            self.param.oversampling_factor,
            is_matched=False,
            length_in_symbols=self.param.filter_length_in_symbols,
            roll_off=self.param.roll_off_factor,
            bandwidth_factor=self.param.bandwidth /
            self.param.symbol_rate)

        if self.param.filter_type == "RAISED_COSINE":
            # for raised cosine, receive filter is a low-pass filter with
            # bandwidth Rs(1+roll-off)/2
            self._filter_rx = ShapingFilter(
                "RAISED_COSINE",
                self.param.oversampling_factor,
                self.param.filter_length_in_symbols,
                0,
                1. + self.param.roll_off_factor)
        else:
            # for all other filter types, receive filter is a matched filter
            self._filter_rx = ShapingFilter(
                self.param.filter_type,
                self.param.oversampling_factor,
                is_matched=True,
                length_in_symbols=self.param.filter_length_in_symbols,
                roll_off=self.param.roll_off_factor,
                bandwidth_factor=self.param.bandwidth /
                self.param.symbol_rate)

    def _set_sampling_indices(self) -> None:
        """ Determines the sampling instants for pilots and data at a given frame
        """
        # create a vector with the position of every pilot and data symbol in a
        # frame
        preamble_symbol_idx = np.arange(
            self.param.number_preamble_symbols) * self._samples_in_pilot
        start_idx = self.param.number_preamble_symbols * self._samples_in_pilot
        self._data_symbol_idx = start_idx + \
            np.arange(self.param.number_data_symbols) * \
            self.param.oversampling_factor
        start_idx += self.param.number_data_symbols * self.param.oversampling_factor
        postamble_symbol_idx = start_idx + \
            np.arange(self.param.number_postamble_symbols) * \
            self._samples_in_pilot
        self._symbol_idx = np.concatenate(
            (preamble_symbol_idx, self._data_symbol_idx, postamble_symbol_idx))

        self._data_symbol_idx += int(self.param.oversampling_factor / 2)
        self._symbol_idx += int(self.param.oversampling_factor / 2)

    def get_bit_energy(self) -> float:
        return 1 / self.param.bits_per_symbol

    def get_symbol_energy(self) -> float:
        return 1.0

    def get_power(self) -> float:
        return 1 / self.param.oversampling_factor
