import unittest
import numpy as np
import numpy.testing as npt

from parameters_parser.parameters_channel import ParametersChannel
from channel.channel import Channel


class TestChannel(unittest.TestCase):

    def setUp(self) -> None:
        self.rnd = np.random.RandomState()
        self.param = ParametersChannel()
        self.param.gain = 1
        self.param.multipath_model = 'NONE'

    def test_propagate(self) -> None:
        """
        Test if the signal at the output of the channel is the same as in the input of the channel.
        Consider different number of transmit and receive antennas.
        """
        signal_length_min = 100
        signal_length_max = 1000

        max_ants = 10

        sampling_rate = 1e6

        ###########
        # test SISO
        channel = Channel(self.param, 1, 1, self.rnd, sampling_rate)

        # with real-valued 1D-array
        signal_length = self.rnd.randint(
            signal_length_min, signal_length_max + 1)
        signal_in = self.rnd.normal(size=signal_length)
        signal_out = channel.propagate(signal_in)
        npt.assert_array_almost_equal(
            np.reshape(signal_in, (1, -1)), signal_out)

        # with complex-valued 2D-array
        signal_length = self.rnd.randint(signal_length_min, signal_length_max)
        signal_in = self.rnd.normal(
            size=(1, signal_length)) + 1j * self.rnd.normal(size=(1, signal_length))
        signal_out = channel.propagate(signal_in)
        npt.assert_array_almost_equal(signal_in, signal_out)

        ############
        # test MIMO

        # test MIMO with Ntx = Nrx
        number_tx_ants = self.rnd.randint(2, max_ants + 1)
        number_rx_ants = number_tx_ants
        channel = Channel(
            self.param,
            number_tx_ants,
            number_rx_ants,
            self.rnd,
            sampling_rate)

        signal_length = self.rnd.randint(signal_length_min, signal_length_max)
        frame_size = (number_tx_ants, signal_length)
        signal_in = self.rnd.normal(
            size=frame_size) + 1j * self.rnd.normal(size=frame_size)
        signal_out = channel.propagate(signal_in)
        npt.assert_array_almost_equal(signal_in, signal_out)

        # test MIMO with Ntx > Nrx
        number_rx_ants = self.rnd.randint(2, max_ants + 1)
        number_tx_ants = number_rx_ants + self.rnd.randint(2, max_ants + 1)
        channel = Channel(
            self.param,
            number_tx_ants,
            number_rx_ants,
            self.rnd,
            sampling_rate)

        signal_length = self.rnd.randint(signal_length_min, signal_length_max)
        frame_size = (number_tx_ants, signal_length)
        signal_in = self.rnd.normal(
            size=frame_size) + 1j * self.rnd.normal(size=frame_size)
        signal_out = channel.propagate(signal_in)
        npt.assert_array_almost_equal(
            signal_in[:number_rx_ants, :], signal_out)

        # test MIMO with Nrx > Ntx
        number_tx_ants = self.rnd.randint(2, max_ants + 1)
        number_rx_ants = number_tx_ants + self.rnd.randint(2, max_ants + 1)
        channel = Channel(
            self.param,
            number_tx_ants,
            number_rx_ants,
            self.rnd,
            sampling_rate)

        signal_length = self.rnd.randint(signal_length_min, signal_length_max)
        frame_size = (number_tx_ants, signal_length)
        signal_in = self.rnd.normal(
            size=frame_size) + 1j * self.rnd.normal(size=frame_size)
        signal_out = channel.propagate(signal_in)
        npt.assert_array_almost_equal(
            signal_in, signal_out[:number_tx_ants, :])
        npt.assert_array_almost_equal(np.zeros((number_rx_ants - number_tx_ants, signal_length)),
                                      signal_out[number_tx_ants:, :])

    def test_impulse_response(self) -> None:
        """
        Test if the impulse response is given correctly with the right number of dimensions.
        """

        max_number_instants = 100
        max_ants = 10
        sampling_rate = 1e6

        # SISO
        channel = Channel(self.param, 1, 1, self.rnd, sampling_rate)
        timestamps = self.rnd.random_sample(
            self.rnd.randint(max_number_instants))
        channel_state_info = channel.get_impulse_response(timestamps)
        npt.assert_array_almost_equal(
            channel_state_info, np.ones(
                (timestamps.size, 1, 1, 1)))

        # MIMO with Ntx = Nrx
        number_tx_ants = self.rnd.randint(2, max_ants + 1)
        number_rx_ants = number_tx_ants
        channel = Channel(
            self.param,
            number_tx_ants,
            number_rx_ants,
            self.rnd,
            sampling_rate)
        timestamps = self.rnd.random_sample(
            self.rnd.randint(max_number_instants))

        channel_state_info = channel.get_impulse_response(timestamps)

        desired_size = (timestamps.size, number_rx_ants, number_tx_ants, 1)
        self.assertEqual(desired_size, channel_state_info.shape)

        for idx_time in range(timestamps.size):
            npt.assert_array_almost_equal(np.eye(number_tx_ants), np.squeeze(
                channel_state_info[idx_time, :, :, :]))

        # MIMO with Ntx > NRx
        number_rx_ants = self.rnd.randint(2, max_ants + 1)
        number_tx_ants = number_rx_ants + self.rnd.randint(2, max_ants + 1)
        channel = Channel(
            self.param,
            number_tx_ants,
            number_rx_ants,
            self.rnd,
            sampling_rate)
        timestamps = self.rnd.random_sample(
            self.rnd.randint(max_number_instants))

        channel_state_info = channel.get_impulse_response(timestamps)

        desired_size = (timestamps.size, number_rx_ants, number_tx_ants, 1)
        self.assertEqual(desired_size, channel_state_info.shape)

        ideal_response = np.concatenate((np.eye(number_rx_ants),
                                         np.zeros((number_rx_ants, number_tx_ants - number_rx_ants))), axis=1)
        for idx_time in range(timestamps.size):
            npt.assert_array_almost_equal(ideal_response, np.squeeze(
                channel_state_info[idx_time, :, :, :]))

        # MIMO with Nrx > Ntx
        number_tx_ants = self.rnd.randint(2, max_ants + 1)
        number_rx_ants = number_tx_ants + self.rnd.randint(2, max_ants + 1)
        channel = Channel(
            self.param,
            number_tx_ants,
            number_rx_ants,
            self.rnd,
            sampling_rate)
        timestamps = self.rnd.random_sample(
            self.rnd.randint(max_number_instants))

        channel_state_info = channel.get_impulse_response(timestamps)

        desired_size = (timestamps.size, number_rx_ants, number_tx_ants, 1)
        self.assertEqual(desired_size, channel_state_info.shape)

        ideal_response = np.concatenate((np.eye(number_tx_ants),
                                         np.zeros((number_rx_ants - number_tx_ants, number_tx_ants))))
        for idx_time in range(timestamps.size):
            npt.assert_array_almost_equal(ideal_response, np.squeeze(
                channel_state_info[idx_time, :, :, :]))

    def test_channel_gain(self) -> None:
        """
        Test if channel gain is applied correctly on both propagation and channel impulse response
        """
        self.param.gain = 2
        sampling_rate = 1e6
        signal_length = 1000

        channel = Channel(self.param, 1, 1, self.rnd, sampling_rate)

        frame_size = (1, signal_length)
        signal_in = self.rnd.normal(
            size=frame_size) + 1j * self.rnd.normal(size=frame_size)
        signal_out = channel.propagate(signal_in)
        npt.assert_array_equal(signal_in * self.param.gain, signal_out)

        timestamps = np.asarray([0, 100, 500]) / sampling_rate
        channel_state_info = channel.get_impulse_response(timestamps)

        npt.assert_array_equal(channel_state_info, self.param.gain)

        self.param.gain = 1


if __name__ == '__main__':
    unittest.main()
