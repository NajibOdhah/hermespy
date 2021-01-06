import unittest
import shutil
import os

import numpy as np
from scipy.io import loadmat

import hermes
import tests.unit_tests.modem.utils as utils
from channel.multipath_fading_channel import MultipathFadingChannel
from source.bits_source import BitsSource
from parameters_parser.parameters_channel import ParametersChannel
from parameters_parser.parameters_ofdm import ParametersOfdm
from modem.waveform_generator_ofdm import WaveformGeneratorOfdm


class TestWaveformGeneratorOfdm(unittest.TestCase):

    def setUp(self) -> None:
        self.rnd = np.random.RandomState(42)
        self.source = BitsSource(self.rnd)

        #######################################################################
        # create an instance of a DL modem with no oversampling
        self.param_dl = ParametersOfdm()

        # create an instance of a DL modem with no oversampling
        self.dl_frame_number_of_dl_slots = 16
        self.dl_frame_number_of_ul_slots = 4

        self.param_dl._link_direction_str = 'dl'

        self.param_dl.subcarrier_spacing = 15000
        self.param_dl.fft_size = 2048
        self.param_dl.oversampling_factor = 1
        self.param_dl.sampling_rate = (self.param_dl.subcarrier_spacing * self.param_dl.fft_size *
                                       self.param_dl.oversampling_factor)

        self.param_dl.number_occupied_subcarriers = 1200
        self.param_dl.guard_interval_ratio = np.asarray([0.0703125, 0.078125])
        self.param_dl.modulation_order = 2
        self.param_dl.precoding = 'none'
        self.param_dl.frame_guard_interval = .001
        self.param_dl.bits_in_ofdm_symbol = (
            self.param_dl.number_occupied_subcarriers
            * int(np.log2(self.param_dl.modulation_order))
        )
        self.dl_frame_symbols_in_slot = 7
        self.dl_frame_reference_symbols = 2
        self.dl_frame_guard_intervals = 2
        self.param_dl._frame_structure_str = ("rr, " + str(self.dl_frame_number_of_dl_slots) + "(c1d cd cd cd cd cd cd), g, "
                                              + str(self.dl_frame_number_of_ul_slots) + "(c1u cu cu cu cu zu zu), g")

        self.param_dl.channel_estimation = "IDEAL"
        self.param_dl.equalization = "ZF"
        self.param_dl._check_params()
        self.dl_modem_no_oversample = WaveformGeneratorOfdm(self.param_dl, self.rnd)

        #######################################################################
        # create an instance of an UL modem with oversampling
        self.param_ul = ParametersOfdm()

        self.ul_frame_number_of_dl_slots = 10
        self.ul_frame_number_of_ul_slots = 10

        self.param_ul._link_direction_str = 'ul'

        self.param_ul.subcarrier_spacing = 7500
        self.param_ul.fft_size = 1024
        self.param_ul.oversampling_factor = 16
        self.param_ul.sampling_rate = (self.param_ul.subcarrier_spacing * self.param_ul.fft_size *
                                       self.param_ul.oversampling_factor)

        self.param_ul.number_occupied_subcarriers = 800
        self.param_ul.guard_interval_ratio = np.asarray([0.0703125, 0.078125])
        self.param_ul.modulation_order = 4
        self.param_ul.precoding = 'none'
        self.param_ul.frame_guard_interval = .001
        self.param_ul.bits_in_ofdm_symbol = (
            self.param_ul.number_occupied_subcarriers
            * int(np.log2(self.param_ul.modulation_order))
        )
        self.ul_frame_symbols_in_slot = 4
        self.ul_frame_reference_symbols = 3
        self.ul_frame_guard_intervals = 1
        self.param_ul._frame_structure_str = ("rrr, " + str(self.ul_frame_number_of_dl_slots) + "(c1d cd cd cd), g, "
                                              + str(self.ul_frame_number_of_ul_slots) + "(c1u cu cu cu)")
        self.ul_frame_cyclic_prefix_overhead = (self.param_ul.guard_interval_ratio[1] +
                                                self.param_ul.guard_interval_ratio[0] * 3) / 4 + 1

        self.param_ul.channel_estimation = "IDEAL"
        self.param_ul.equalization = "ZF"
        self.param_ul._check_params()

        self.ul_modem_oversample = WaveformGeneratorOfdm(self.param_ul, self.rnd)

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ["MPLBACKEND"] = ""

    def test_number_of_bits_in_frame(self) -> None:
        """
        Test if frame is created with the right number of data bits
        """
        bits_in_frame_dl = (self.dl_frame_symbols_in_slot * self.dl_frame_number_of_dl_slots *
                            np.log2(self.param_dl.modulation_order) * self.param_dl.number_occupied_subcarriers)
        self.assertEqual(bits_in_frame_dl, self.param_dl.bits_in_frame)

        bits_in_frame_ul = (self.ul_frame_symbols_in_slot * self.ul_frame_number_of_ul_slots *
                            np.log2(self.param_ul.modulation_order) * self.param_ul.number_occupied_subcarriers)
        self.assertEqual(bits_in_frame_ul, self.param_ul.bits_in_frame)

    def test_MMSE_lower_BER_than_ZF(self) -> None:
        """Checks if MMSE is actually performed by checking if BER is lower for
        low SNRs than for ZF equalization. """
        settings_dir_ZF = os.path.join(
            "tests", "unit_tests", "modem", "res", "settings_ZF")
        settings_dir_MMSE = os.path.join(
            "tests", "unit_tests", "modem", "res", "settings_MMSE")

        results_dir = os.path.join(settings_dir_ZF, "..", "results")
        arguments_hermes = ["-p", settings_dir_ZF, "-o", results_dir]

        import matplotlib.pyplot as plt
        plt.switch_backend("agg")
        hermes.hermes(arguments_hermes)

        # get ZF results at first
        results_simulation_ZF = loadmat(
            os.path.join(results_dir, "statistics.mat"))
        shutil.rmtree(results_dir)

        # now MMSE results
        arguments_hermes = ["-p", settings_dir_MMSE, "-o", results_dir]
        hermes.hermes(arguments_hermes)
        results_simulation_MMSE = loadmat(
            os.path.join(results_dir, "statistics.mat"))
        shutil.rmtree(results_dir)

        # BER of MMSE should be lower for low SNRs than ZF
        ber_mean_ZF = results_simulation_ZF["ber_mean"]
        ber_mean_MMSE = results_simulation_MMSE["ber_mean"]

        np.testing.assert_array_less(ber_mean_MMSE, ber_mean_ZF)

    def test_dl_frame_creation_with_proper_size(self) -> None:
        """
        Test if downlink frame is created with the right size and right number of bits
        """

        samples_in_short_prefix = int(
            round(
                self.param_dl.guard_interval_ratio[0] *
                self.param_dl.fft_size))
        samples_in_long_prefix = int(
            round(
                self.param_dl.guard_interval_ratio[1] *
                self.param_dl.fft_size))
        samples_in_slot = (self.dl_frame_symbols_in_slot * self.param_dl.fft_size + samples_in_long_prefix +
                           (self.dl_frame_symbols_in_slot - 1) * samples_in_short_prefix)

        samples_in_reference = self.dl_frame_reference_symbols * self.param_dl.fft_size
        samples_in_dl = self.dl_frame_number_of_dl_slots * samples_in_slot
        samples_in_ul = self.dl_frame_number_of_ul_slots * samples_in_slot
        samples_in_guard = int(round(self.param_dl.frame_guard_interval * self.param_dl.subcarrier_spacing
                                     * self.param_dl.fft_size))

        samples_in_frame = (samples_in_reference + samples_in_dl + samples_in_ul +
                            self.dl_frame_guard_intervals * samples_in_guard)
        samples_in_frame *= self.param_dl.oversampling_factor

        initial_timestamp = 1000
        bits_in_frame = utils.flatten_blocks(
            self.source.get_bits(self.param_dl.bits_in_frame))
        signal, timestamp_final, initial_sample = self.dl_modem_no_oversample.create_frame(
            initial_timestamp, bits_in_frame)

        self.assertEqual(initial_sample, initial_timestamp)
        self.assertEqual(timestamp_final, initial_timestamp + samples_in_frame)

        self.assertEqual(
            samples_in_frame,
            self.dl_modem_no_oversample.samples_in_frame)
        self.assertEqual(samples_in_frame, signal.size)

    def test_ul_frame_creation_with_proper_size(self) -> None:
        """
        Test if uplink frame is created with the right size and right number of bits
        """

        samples_in_short_prefix = int(
            round(
                self.param_ul.guard_interval_ratio[0] *
                self.param_ul.fft_size))
        samples_in_long_prefix = int(
            round(
                self.param_ul.guard_interval_ratio[1] *
                self.param_ul.fft_size))
        samples_in_slot = (self.ul_frame_symbols_in_slot * self.param_ul.fft_size + samples_in_long_prefix +
                           (self.ul_frame_symbols_in_slot - 1) * samples_in_short_prefix)

        samples_in_frame = self.ul_frame_reference_symbols * \
            self.param_ul.fft_size  # reference signal
        samples_in_frame += (self.ul_frame_number_of_dl_slots +
                             self.ul_frame_number_of_ul_slots) * samples_in_slot
        samples_in_guard = int(round(self.param_ul.frame_guard_interval * self.param_ul.subcarrier_spacing
                                     * self.param_ul.fft_size))  # guard intervals
        samples_in_frame += self.ul_frame_guard_intervals * samples_in_guard
        samples_in_frame *= self.param_ul.oversampling_factor

        initial_timestamp = 1000
        bits_in_frame = utils.flatten_blocks(
            self.source.get_bits(self.param_ul.bits_in_frame))

        signal, timestamp_final, initial_sample = self.ul_modem_oversample.create_frame(
            initial_timestamp, bits_in_frame)

        self.assertEqual(initial_sample, initial_timestamp)
        self.assertEqual(timestamp_final, initial_timestamp + samples_in_frame)

        self.assertEqual(
            samples_in_frame,
            self.ul_modem_oversample.samples_in_frame)
        self.assertEqual(samples_in_frame, signal.size)

    def test_zeros_in_guard_interval(self) -> None:
        """
        Test if frame is created with zeros at the guard interval
        """

        samples_in_short_prefix = int(
            round(
                self.param_dl.guard_interval_ratio[0] *
                self.param_dl.fft_size))
        samples_in_long_prefix = int(
            round(
                self.param_dl.guard_interval_ratio[1] *
                self.param_dl.fft_size))
        samples_in_slot = (self.dl_frame_symbols_in_slot * self.param_dl.fft_size + samples_in_long_prefix +
                           (self.dl_frame_symbols_in_slot - 1) * samples_in_short_prefix)

        samples_in_reference = 2 * self.param_dl.fft_size
        samples_in_dl = self.dl_frame_number_of_dl_slots * samples_in_slot
        samples_in_ul = self.dl_frame_number_of_ul_slots * samples_in_slot

        samples_in_guard = int(round(self.param_dl.frame_guard_interval * self.param_dl.subcarrier_spacing
                                     * self.param_dl.fft_size))

        samples_in_frame = samples_in_reference + \
            samples_in_dl + samples_in_ul + 2 * samples_in_guard

        bits_in_frame = utils.flatten_blocks(
            self.source.get_bits(self.param_dl.bits_in_frame))
        signal, timestamp_final, initial_sample = self.dl_modem_no_oversample.create_frame(
            0, bits_in_frame)

        initial_zero_index = samples_in_reference + samples_in_dl
        np.testing.assert_almost_equal(signal[initial_zero_index:samples_in_frame],
                                       np.zeros(samples_in_frame - initial_zero_index))

    def test_zero_forcing_equalization_with_cyclic_prefix(self) -> None:
        """
        Test zero-forcing equalization with cyclic prefix
        """
        number_of_symbols = 10
        number_of_paths = 10

        for idx in range(number_of_symbols):
            cyclic_prefix_samples = number_of_paths

            resource_elements = (self.rnd.normal(size=self.param_dl.number_occupied_subcarriers) +
                                 1j * self.rnd.normal(size=self.param_dl.number_occupied_subcarriers))
            ofdm_symbol = self.dl_modem_no_oversample.generate_ofdm_symbol(
                resource_elements, cyclic_prefix_samples)

            channel = self.rnd.normal(
                size=number_of_paths) + 1j * self.rnd.normal(size=number_of_paths)
            channel_in_freq_domain = np.fft.fft(
                channel, n=self.param_dl.fft_size)

            rx_symbol = np.convolve(channel, ofdm_symbol)
            rx_symbol = rx_symbol[cyclic_prefix_samples:
                                  cyclic_prefix_samples + self.param_dl.fft_size]

            detected_symbol = self.dl_modem_no_oversample.demodulate_ofdm_symbol(
                rx_symbol, channel_in_freq_domain, 0)

            np.testing.assert_almost_equal(resource_elements, detected_symbol)

    def test_channel_estimation_ideal(self) -> None:
        """
        Test ideal channel estimation for a SISO system.
        In this test we verify if the 'WaveformGeneratorOfdm.channel_estimation' method returns the expected frequency
        response from a known channel.
        """
        # create a channel
        channel_param = ParametersChannel()
        channel_param.multipath_model = "EXPONENTIAL"
        channel_param.tap_interval = 1 / self.param_ul.sampling_rate
        channel_param.rms_delay = 1 / self.param_ul.subcarrier_spacing * \
            self.param_ul.guard_interval_ratio[1] / 30
        channel_param.velocity = np.asarray([10, 0, 0])
        channel_param.attenuation_db = 0

        channel_param.check_params()

        channel = MultipathFadingChannel(
            channel_param,
            1,
            1,
            np.random.RandomState(),
            self.param_ul.sampling_rate,
            50)
        channel.init_drop()

        ul_frame_symbols_idxs = np.asarray([])
        idx = 0
        for element in self.param_ul.frame_structure:
            if element.type == ParametersOfdm.element_types.UPLINK:
                ul_frame_symbols_idxs = np.append(ul_frame_symbols_idxs, idx)
            idx += element.number_of_samples

        ul_frame_symbols_idxs *= self.param_ul.oversampling_factor

        sampled_channel = channel.get_impulse_response(
            ul_frame_symbols_idxs / self.param_ul.sampling_rate)

        long_fft_size = self.ul_modem_oversample.param.fft_size * \
            self.param_ul.oversampling_factor
        expected_channel_in_frequency = np.fft.fft(
            sampled_channel, n=long_fft_size)
        remove_idx = np.arange(int(self.ul_modem_oversample.param.fft_size / 2),
                               int(long_fft_size - self.ul_modem_oversample.param.fft_size / 2))
        expected_channel_in_frequency = np.transpose(
            np.squeeze(expected_channel_in_frequency))
        expected_channel_in_frequency = np.delete(
            expected_channel_in_frequency, remove_idx, 0)

        self.ul_modem_oversample.set_channel(channel)
        self.ul_modem_oversample.param.channel_estimation = 'IDEAL'
        estimated_channel = self.ul_modem_oversample.channel_estimation(
            None, 0)

        np.testing.assert_allclose(
            expected_channel_in_frequency,
            estimated_channel)

    def test_channel_estimation_ideal_preamble(self) -> None:
        """
        Test ideal preamble-based channel estimation for a SISO system.
        In this test we verify if the 'WaveformGeneratorOfdm.channel_estimation' method returns the expected frequency
        response from a known channel at the beginning of a frame.
        """
        self._test_channel_estimation_ideal_reference("IDEAL_PREAMBLE")

    def test_channel_estimation_ideal_postamble(self) -> None:
        """
        Test ideal preamble-based channel estimation for a SISO system.
        In this test we verify if the 'WaveformGeneratorOfdm.channel_estimation' method returns the expected frequency
        response from a known channel at the end of a frame.
        """
        self._test_channel_estimation_ideal_reference("IDEAL_POSTAMBLE")

    def test_channel_estimation_ideal_midamble(self) -> None:
        """
        Test ideal preamble-based channel estimation for a SISO system.
        In this test we verify if the 'WaveformGeneratorOfdm.channel_estimation' method returns the expected frequency
        response from a known channel in the middle of a frame.
        """
        self._test_channel_estimation_ideal_reference("IDEAL_MIDAMBLE")

    def _test_channel_estimation_ideal_reference(
            self, position_in_frame: str) -> None:
        """
        Test ideal reference-signal-based channel estimation for a single reference in an UL frame.
        In this test we verify if the 'WaveformGeneratorOfdm.channel_estimation' method returns the expected frequency
        response from a known channel at a given position in the frame.

        Args:
            position_in_frame(str): indicates at which point the reference signal was considered. The following values
                are accepted:
                    "IDEAL_PREAMBLE" - reference signal is at the beginning of the frame
                    "IDEAL_POSTAMBLE" - reference signal is at the end of the frame
                    "IDEAL_MIDAMBLE" - reference signal is exactly in thevmiddle of the frame
        """

        # create a channel
        channel_param = ParametersChannel()
        channel_param.multipath_model = "COST259"
        channel_param.cost_259_type = "TYPICAL_URBAN"
        channel_param.attenuation_db = 3
        channel_param.velocity = np.asarray([np.random.random() * 20, 0, 0])

        channel_param.check_params()

        channel = MultipathFadingChannel(
            channel_param,
            1,
            1,
            np.random.RandomState(),
            self.param_ul.sampling_rate,
            50)
        channel.init_drop()

        if position_in_frame == "IDEAL_PREAMBLE":
            timestamp = 0.
        elif position_in_frame == "IDEAL_POSTAMBLE":
            timestamp = self.ul_modem_oversample.samples_in_frame / \
                self.ul_modem_oversample.param.sampling_rate
        elif position_in_frame == "IDEAL_MIDAMBLE":
            timestamp = self.ul_modem_oversample.samples_in_frame / \
                self.ul_modem_oversample.param.sampling_rate * .5
        else:
            raise ValueError("invalid 'position_in_frame'")

        sampled_channel = channel.get_impulse_response(np.asarray([timestamp]))

        long_fft_size = self.ul_modem_oversample.param.fft_size * \
            self.param_ul.oversampling_factor
        expected_channel_in_frequency = np.fft.fft(
            sampled_channel, n=long_fft_size)
        remove_idx = np.arange(int(self.ul_modem_oversample.param.fft_size / 2),
                               int(long_fft_size - self.ul_modem_oversample.param.fft_size / 2))
        expected_channel_in_frequency = np.transpose(
            np.squeeze(expected_channel_in_frequency, axis=(1, 2)))
        expected_channel_in_frequency = np.delete(
            expected_channel_in_frequency, remove_idx, 0)

        self.ul_modem_oversample.set_channel(channel)
        self.ul_modem_oversample.param.channel_estimation = position_in_frame
        estimated_channel = self.ul_modem_oversample.channel_estimation(
            None, 0)
        number_of_channel_samples = estimated_channel.shape[1]
        expected_channel_in_frequency = np.tile(
            expected_channel_in_frequency, number_of_channel_samples)

        np.testing.assert_allclose(
            expected_channel_in_frequency,
            estimated_channel)


if __name__ == '__main__':
    unittest.main()
