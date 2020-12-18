import unittest
import numpy as np
import numpy.testing as npt
from scipy import stats
from scipy import special
from matplotlib import pyplot as plt
import copy

from parameters_parser.parameters_channel import ParametersChannel
from channel.multipath_fading_channel import MultipathFadingChannel


class TestMultipathFadingChannel(unittest.TestCase):

    def setUp(self) -> None:
        self.rnd = np.random.RandomState(42)
        self.param = ParametersChannel()
        self.param.multipath_model = 'STOCHASTIC'
        self.param.gain = 1

    def test_propagation_siso_no_fading(self) -> None:
        """
        Test the propagation through a SIDO multipath channel model without fading
        Check if the output sizes are consistent
        Check the output of a SISO multipath channel model without fading (K factor of Rice distribution = inf)
        """
        max_number_paths = 20
        max_delay_in_samples = 30
        max_doppler_frequency = 100
        min_number_samples = 1000
        max_number_samples = 500000

        sampling_rate = 1e6

        number_of_paths = self.rnd.randint(2, max_number_paths)
        path_delay_in_samples = self.rnd.choice(
            max_delay_in_samples, number_of_paths, replace=False)
        path_delay_in_samples = np.sort(
            path_delay_in_samples) - path_delay_in_samples.min()
        max_delay_in_samples = path_delay_in_samples.max()
        path_delay = path_delay_in_samples / sampling_rate
        self.param.delays = path_delay

        power_delay_profile = np.abs(
            self.rnd.random_sample(
                self.param.delays.shape))
        self.param.power_delay_profile = power_delay_profile / \
            np.sum(power_delay_profile)
        doppler_frequency = self.rnd.random_sample() * max_doppler_frequency

        self.param.k_factor_rice = np.inf * \
            np.ones(path_delay.size)  # only LOS

        channel = MultipathFadingChannel(
            self.param, 1, 1, self.rnd, sampling_rate, doppler_frequency)
        channel.init_drop()

        number_of_samples = self.rnd.randint(
            min_number_samples, max_number_samples)
        signal_in = np.zeros(number_of_samples)
        impulse_interval = max_delay_in_samples + 1
        impulse_idx = np.arange(number_of_samples, step=impulse_interval)
        signal_in[impulse_idx] = 1

        output = channel.propagate(signal_in)
        self.assertEqual(
            output.shape,
            (1,
             number_of_samples +
             max_delay_in_samples))

        desired_output = np.zeros(output.shape, dtype=float)
        for path_idx in range(number_of_paths):
            desired_output[0, impulse_idx + path_delay_in_samples[path_idx]] = \
                np.sqrt(self.param.power_delay_profile[path_idx])

        npt.assert_array_almost_equal(desired_output, np.abs(output))

    def test_propagation_fading(self) -> None:
        """
        Test the propagation through a SIDO multipath channel with fading
        Compare the output to the expected impulse response in a time-variant channel
        Test if impulse response is returned correctly
        """
        min_number_samples = 1000
        max_number_samples = 500000

        max_number_paths = 20
        max_delay_in_samples = 30
        max_doppler_frequency = 100

        sampling_rate = 1e6

        number_of_paths = self.rnd.randint(
            2, np.minimum(max_number_paths, max_delay_in_samples))
        path_delay_in_samples = self.rnd.choice(
            max_delay_in_samples, number_of_paths, replace=False)
        path_delay_in_samples = np.sort(
            path_delay_in_samples) - path_delay_in_samples.min()
        max_delay_in_samples = path_delay_in_samples.max()
        path_delay = path_delay_in_samples / sampling_rate
        self.param.delays = path_delay

        power_delay_profile = np.abs(
            self.rnd.random_sample(
                self.param.delays.shape))
        self.param.power_delay_profile = power_delay_profile / \
            np.sum(power_delay_profile)

        doppler_frequency = self.rnd.random_sample() * max_doppler_frequency

        self.param.k_factor_rice = np.zeros(path_delay.size)  # Rayleigh fading

        channel = MultipathFadingChannel(
            self.param, 1, 1, self.rnd, sampling_rate, doppler_frequency)
        channel.init_drop()

        number_of_samples = self.rnd.randint(
            min_number_samples, max_number_samples)
        signal_in = np.zeros(number_of_samples)
        impulse_interval = max_delay_in_samples + 1
        sampling_indices = np.arange(number_of_samples, step=impulse_interval)
        signal_in[sampling_indices] = 1

        output = channel.propagate(signal_in)

        timestamps = sampling_indices / sampling_rate
        channel_impulse_response = channel.get_impulse_response(timestamps)
        self.assertEqual(channel_impulse_response.shape,
                         (timestamps.size, 1, 1, max_delay_in_samples + 1))

        desired_output = np.zeros(output.shape, dtype=complex)
        for path_idx in range(number_of_paths):
            desired_output[0, sampling_indices + path_delay_in_samples[path_idx]] = \
                channel_impulse_response[:, 0, 0,
                                         path_delay_in_samples[path_idx]]

        npt.assert_array_almost_equal(desired_output, output)

    def test_propagation_fading_interpolation(self) -> None:
        """
        Test the propagation through a SIDO multipath channel with fading and interpolation, i.e., when the paths do not
        fall in ths sampling instants.
        Compare the output to the expected impulse response in a time-variant channel
        """
        min_number_samples = 1000
        max_number_samples = 500000

        max_number_paths = 20
        max_delay_in_samples = 30
        max_doppler_frequency = 100

        sampling_rate = 1e6

        number_of_paths = self.rnd.randint(2, max_number_paths)
        path_delay = self.rnd.random_sample(
            number_of_paths) * max_delay_in_samples / sampling_rate
        path_delay = np.sort(path_delay)
        self.param.delays = path_delay

        power_delay_profile = np.abs(
            self.rnd.random_sample(
                self.param.delays.shape))
        self.param.power_delay_profile = power_delay_profile / \
            np.sum(power_delay_profile)

        doppler_frequency = self.rnd.random_sample() * max_doppler_frequency

        self.param.k_factor_rice = np.zeros(path_delay.size)  # Rayleigh fading

        channel = MultipathFadingChannel(
            self.param, 1, 1, self.rnd, sampling_rate, doppler_frequency)
        max_delay_in_samples = channel.max_delay_in_samples
        channel.init_drop()

        number_of_samples = self.rnd.randint(
            min_number_samples, max_number_samples)
        signal_in = np.zeros(number_of_samples)
        impulse_interval = max_delay_in_samples + 1

        sampling_indices = np.arange(number_of_samples, step=impulse_interval)
        signal_in[sampling_indices] = 1

        output = channel.propagate(signal_in)

        timestamps = sampling_indices / sampling_rate
        channel_impulse_response = channel.get_impulse_response(timestamps)
        self.assertEqual(channel_impulse_response.shape,
                         (timestamps.size, 1, 1, max_delay_in_samples + 1))

        desired_output = np.zeros(output.shape, dtype=complex)
        for path_idx in range(max_delay_in_samples + 1):
            desired_output[0, sampling_indices +
                           path_idx] = channel_impulse_response[:, 0, 0, path_idx]

        npt.assert_array_almost_equal(desired_output, output)

    def test_rayleigh(self) -> None:
        """
        Test if the amplitude of a path is Rayleigh distributed.
        Verify that both real and imaginary components are zero-mean normal random variables with the right variance and
        uncorrelated.
        """
        max_number_of_drops = 200
        doppler_frequency = 200
        sampling_rate = 1e6
        samples_per_drop = 1000

        self.param.delays = np.array([0])
        self.param.power_delay_profile = np.array([1])
        self.param.k_factor_rice = np.asarray([0])

        channel = MultipathFadingChannel(
            self.param, 1, 1, self.rnd, sampling_rate, doppler_frequency)
        time_interval = 1 / doppler_frequency  # get uncorrelated samples
        timestamps = np.arange(samples_per_drop) * time_interval

        samples = np.array([])

        is_rayleigh = False
        alpha = .05
        max_corr = 0.05

        aux_samples = np.array([])

        number_of_drops = 0

        while not is_rayleigh and number_of_drops < max_number_of_drops:
            channel.init_drop()
            channel_gains = channel.get_impulse_response(timestamps)
            samples = np.append(samples, channel_gains.ravel())
            aux_samples = np.append(
                aux_samples, self.rnd.randn(
                    timestamps.size))

            dummy, p_real = stats.kstest(
                np.real(samples), 'norm', args=(
                    0, 1 / np.sqrt(2)))
            dummy, p_imag = stats.kstest(
                np.real(samples), 'norm', args=(
                    0, 1 / np.sqrt(2)))

            corr = np.corrcoef(np.real(samples), np.imag(samples))
            corr = corr[0, 1]

            if p_real > alpha and p_imag > alpha and np.abs(corr) < max_corr:
                is_rayleigh = True

            number_of_drops += 1

        self.assertTrue(is_rayleigh)

    def test_rice(self) -> None:
        """
        Test if the amplitude of a path is Ricean distributed.
        """
        max_number_of_drops = 100
        doppler_frequency = 200
        sampling_rate = 1e6
        samples_per_drop = 1000

        self.param.delays = np.array([0])
        self.param.power_delay_profile = np.array([1])
        self.param.k_factor_rice = np.asarray([1])

        channel = MultipathFadingChannel(
            self.param, 1, 1, self.rnd, sampling_rate, doppler_frequency)
        time_interval = 1 / doppler_frequency  # get uncorrelated samples
        timestamps = np.arange(samples_per_drop) * time_interval

        samples = np.array([])

        is_rice = False
        alpha = .05

        aux_samples = np.array([])
        number_of_drops = 0
        while not is_rice and number_of_drops < max_number_of_drops:
            channel.init_drop()
            channel_gains = channel.get_impulse_response(timestamps)
            samples = np.append(samples, channel_gains.ravel())
            aux_samples = np.append(
                aux_samples, self.rnd.randn(
                    timestamps.size))

            dummy, p_real = stats.kstest(
                np.abs(samples), 'rice', args=(
                    np.sqrt(2), 0, 1 / 2))

            if p_real > alpha:
                is_rice = True

            number_of_drops += 1

        self.assertTrue(is_rice)

    def test_power_delay_profile(self) -> None:
        """
        Test if the resulting power delay profile matches with the one specified in the parameters.
        Test also an interpolated channel (should have the same rms delay spread)
        """
        max_number_paths = 20
        max_delay_in_samples = 30
        max_doppler_frequency = 100

        max_number_of_drops = 100
        samples_per_drop = 1000

        sampling_rate = 1e6

        max_delay_spread_dev = sampling_rate / 10  # maximum deviation of delay spread

        number_of_paths = self.rnd.randint(
            2, np.minimum(max_number_paths, max_delay_in_samples))
        path_delay_in_samples = self.rnd.choice(
            max_delay_in_samples, number_of_paths, replace=False)
        path_delay_in_samples = np.sort(
            path_delay_in_samples) - path_delay_in_samples.min()
        max_delay_in_samples = path_delay_in_samples.max()
        path_delay = path_delay_in_samples / sampling_rate
        self.param.delays = path_delay

        power_delay_profile = np.abs(
            self.rnd.random_sample(
                self.param.delays.shape))
        self.param.power_delay_profile = power_delay_profile / \
            np.sum(power_delay_profile)

        doppler_frequency = self.rnd.random_sample() * max_doppler_frequency

        self.param.k_factor_rice = np.zeros(path_delay.size)  # Rayleigh fading

        channel = MultipathFadingChannel(
            self.param, 1, 1, self.rnd, sampling_rate, doppler_frequency)

        # create channel with interpolation (delays are not multiples of
        # sampling interval)
        sampling_rate_interp = sampling_rate * \
            (1 + self.rnd.random_sample() * 0.5)
        channel_interp = MultipathFadingChannel(
            self.param, 1, 1, self.rnd, sampling_rate_interp, doppler_frequency)

        time_interval = 1 / doppler_frequency  # get uncorrelated samples
        timestamps = np.arange(samples_per_drop) * time_interval

        number_of_drops = 0
        is_right_profile = False
        is_same_delay_spread = False

        all_channels = None
        all_channels_interp = None

        right_profile = np.zeros(max_delay_in_samples + 1)
        right_profile[path_delay_in_samples] = self.param.power_delay_profile

        while ((not is_right_profile) or (not is_same_delay_spread)
               ) and number_of_drops < max_number_of_drops:
            channel.init_drop()
            channel_interp.init_drop()

            channel_gains = channel.get_impulse_response(timestamps)
            channel_gains = np.squeeze(channel_gains)

            if all_channels is None:
                all_channels = channel_gains
            else:
                all_channels = np.append(all_channels, channel_gains, axis=0)
            mean_power = np.mean(np.abs(all_channels)**2, axis=0)

            rms_delay_spread = np.sqrt(np.sum((np.arange(channel.max_delay_in_samples + 1) / channel.sampling_rate)**2
                                              * mean_power))
            if np.allclose(mean_power, right_profile, atol=1e-3):
                is_right_profile = True

            channel_gains = channel_interp.get_impulse_response(timestamps)
            channel_gains = np.squeeze(channel_gains)

            if all_channels_interp is None:
                all_channels_interp = channel_gains
            else:
                all_channels_interp = np.append(
                    all_channels_interp, channel_gains, axis=0)

            mean_power_interp = np.mean(np.abs(all_channels_interp)**2, axis=0)
            rms_delay_spread_interp = np.sqrt(np.sum((np.arange(channel_interp.max_delay_in_samples + 1)
                                                      / channel_interp.sampling_rate)**2 * mean_power_interp))

            if np.abs(rms_delay_spread -
                      rms_delay_spread_interp) < max_delay_spread_dev:
                is_same_delay_spread = True

        self.assertTrue(is_right_profile and is_same_delay_spread)

    def test_channel_gain(self) -> None:
        """
        Test if channel gain is applied correctly on both propagation and channel impulse response
        """
        gain = 10

        doppler_frequency = 200
        sampling_rate = 1e6
        signal_length = 1000

        self.param.delays = np.array([0])
        self.param.power_delay_profile = np.array([1])
        self.param.k_factor_rice = np.asarray([0])

        param_with_gain = copy.deepcopy(self.param)
        param_with_gain.gain = gain

        rnd_gain = copy.deepcopy(self.rnd)

        channel_no_gain = MultipathFadingChannel(
            self.param, 1, 1, self.rnd, sampling_rate, doppler_frequency)
        channel_no_gain.init_drop()

        channel_gain = MultipathFadingChannel(
            param_with_gain, 1, 1, rnd_gain, sampling_rate, doppler_frequency)
        channel_gain.init_drop()

        frame_size = (1, signal_length)
        signal_in = self.rnd.normal(
            size=frame_size) + 1j * self.rnd.normal(size=frame_size)
        signal_out_no_gain = channel_no_gain.propagate(signal_in)
        signal_out_gain = channel_gain.propagate(signal_in)

        npt.assert_array_equal(signal_out_gain, signal_out_no_gain * gain)

        timestamps = np.asarray([0, 100, 500]) / sampling_rate
        channel_state_info_no_gain = channel_no_gain.get_impulse_response(
            timestamps)
        channel_state_info_gain = channel_gain.get_impulse_response(timestamps)

        npt.assert_array_equal(
            channel_state_info_gain,
            channel_state_info_no_gain * gain)

    def test_mimo(self) -> None:
        """
        Test the MIMO channel
        The following tests are performed

        1 - check output sizes
        2 - test MIMO output with known channel
        3 - check the channel impulse response
        4 - test antenna correlation
        :return:
        """
        pass


if __name__ == '__main__':
    '''
    If this file is called, initially plot some graphs to that indicate if multipath channel is working correctly, and
    then perform all unit tests.
    '''

    rnd = np.random.RandomState(44)
    param = ParametersChannel()
    param.multipath_model = 'STOCHASTIC'

    ###################################
    # plot Rayleigh/Rice distribution
    max_number_of_drops = 100
    doppler_frequency = 200
    sampling_rate = 1e6
    samples_per_drop = 1000

    param.delays = np.array([0])
    param.power_delay_profile = np.array([1])

    param.k_factor_rice = np.asarray([0])
    channel_rayleigh = MultipathFadingChannel(
        param, 1, 1, rnd, sampling_rate, doppler_frequency)

    param.k_factor_rice = np.asarray([1])
    channel_rice = MultipathFadingChannel(
        param, 1, 1, rnd, sampling_rate, doppler_frequency)

    time_interval = 1 / doppler_frequency  # get uncorrelated samples
    timestamps = np.arange(samples_per_drop) * time_interval

    samples_rayleigh = np.array([])
    samples_rice = np.array([])

    aux_samples = np.array([])

    for drop in range(max_number_of_drops):
        channel_rayleigh.init_drop()
        channel_rice.init_drop()

        channel_gains = channel_rayleigh.get_impulse_response(timestamps)
        samples_rayleigh = np.append(samples_rayleigh, channel_gains.ravel())

        channel_gains = channel_rice.get_impulse_response(timestamps)
        samples_rice = np.append(samples_rice, channel_gains.ravel())

    # plot distribution of real part
    plt.figure()
    plt.subplot(131)
    hist, bin_edges, dummy = plt.hist(
        np.real(samples_rayleigh), density=True, bins=40, label='histogram')
    plt.plot(
        bin_edges,
        stats.norm.pdf(
            bin_edges,
            scale=1 /
            np.sqrt(2)),
        label='normal distribution')
    plt.title('Histogram of real part of channel gain')
    plt.legend()

    # plot distribution of absolute value (Rayleigh fading)
    plt.subplot(132)
    hist, bin_edges, dummy = plt.hist(
        np.abs(samples_rayleigh), density=True, bins=40, label='histogram')
    plt.plot(
        bin_edges,
        stats.rayleigh.pdf(
            bin_edges,
            scale=1 /
            np.sqrt(2)),
        label='Rayleigh distribution')
    plt.title('Histogram of channel gain amplitude - Rayleigh fading')
    plt.legend()

    # plot distribution of absolute value (Rice fading)
    plt.subplot(133)
    hist, bin_edges, dummy = plt.hist(
        np.abs(samples_rice), density=True, bins=40, label='histogram')
    plt.plot(
        bin_edges,
        stats.rice.pdf(
            bin_edges,
            b=np.sqrt(2),
            scale=1 / 2),
        label='Rice distribution')
    plt.title('Histogram of channel gain amplitude - Rice fading')
    plt.legend()

    ##########################################################################
    # plot power delay/profile
    max_number_paths = 20
    max_delay_in_samples = 30
    max_doppler_frequency = 100

    number_of_paths = rnd.randint(2, np.minimum(
        max_number_paths, max_delay_in_samples))
    path_delay_in_samples = rnd.choice(
        max_delay_in_samples,
        number_of_paths,
        replace=False)
    path_delay_in_samples = np.sort(
        path_delay_in_samples) - path_delay_in_samples.min()
    max_delay_in_samples = path_delay_in_samples.max()
    path_delay = path_delay_in_samples / sampling_rate

    param.delays = path_delay

    power_delay_profile = np.abs(rnd.random_sample(param.delays.shape))
    param.power_delay_profile = power_delay_profile / \
        np.sum(power_delay_profile)

    doppler_frequency = rnd.random_sample() * max_doppler_frequency

    param.k_factor_rice = np.zeros(path_delay.size)  # Rayleigh fading

    # create channel with no need for interpolation
    channel = MultipathFadingChannel(
        param, 1, 1, rnd, sampling_rate, doppler_frequency)

    # create channel with interpolation (delays are not multiples of sampling
    # interval)
    sampling_rate_interp = sampling_rate * (1 + rnd.random_sample() * 0.5)
    channel_interp = MultipathFadingChannel(
        param, 1, 1, rnd, sampling_rate_interp, doppler_frequency)

    time_interval = 1 / doppler_frequency  # get uncorrelated samples
    timestamps = np.arange(samples_per_drop) * time_interval

    all_channels = np.array([])
    all_channels_interp = np.array([])

    for drop in range(max_number_of_drops):
        channel.init_drop()
        channel_interp.init_drop()

        channel_gains = np.squeeze(channel.get_impulse_response(timestamps))
        if all_channels.size:
            all_channels = np.append(all_channels, channel_gains, axis=0)
        else:
            all_channels = channel_gains

        channel_gains = np.squeeze(
            channel_interp.get_impulse_response(timestamps))
        if all_channels_interp.size:
            all_channels_interp = np.append(
                all_channels_interp, channel_gains, axis=0)
        else:
            all_channels_interp = channel_gains

    mean_power = np.mean(np.abs(all_channels) ** 2, axis=0)
    mean_power_interp = np.mean(np.abs(all_channels_interp) ** 2, axis=0)

    rms_delay_spread = np.sqrt(np.sum((np.arange(channel.max_delay_in_samples + 1) / channel.sampling_rate)**2
                                      * mean_power))
    rms_delay_spread_interp = np.sqrt(np.sum((np.arange(channel_interp.max_delay_in_samples + 1)
                                              / channel_interp.sampling_rate)**2 * mean_power_interp))

    plt.figure()
    plt.subplot(121)
    plt.stem(
        param.delays * 1e6,
        param.power_delay_profile,
        label='desired profile',
        linefmt='r',
        markerfmt='or')
    marker = plt.stem(np.arange(max_delay_in_samples + 1) / sampling_rate * 1e6, mean_power, label='actual profile',
                      linefmt='b', markerfmt='Db')
    marker[0].set_markerfacecolor('none')
    plt.xlabel('delay (ns)')
    plt.title(
        'Power Delay Profile (rms delay spread = {:f}ns)'.format(
            rms_delay_spread * 1e6))
    plt.legend()

    plt.subplot(122)

    plt.stem(
        param.delays * 1e6,
        param.power_delay_profile,
        label='desired profile',
        linefmt='r',
        markerfmt='or')
    marker = plt.stem(np.arange(channel_interp.max_delay_in_samples + 1) / sampling_rate_interp * 1e6, mean_power_interp,
                      label='actual profile', linefmt='b', markerfmt='Db')
    marker[0].set_markerfacecolor('none')
    plt.xlabel('delay (ns)')
    plt.title(
        'Power Delay Profile - Interpolated (rms delay spread = {:f}ns)'.format(
            rms_delay_spread_interp * 1e6))
    plt.legend()

    ##########################################################################
    # plot autocorrelation and Doppler spectrum
    max_number_of_drops = 1000
    doppler_frequency = 200
    sampling_rate = 1e6
    samples_per_drop = 1000
    channel_oversampling_rate = 100  # in relation to doppler_frequency

    param.delays = np.array([0])
    param.power_delay_profile = np.array([1])

    param.k_factor_rice = np.asarray([0])
    channel = MultipathFadingChannel(
        param, 1, 1, rnd, sampling_rate, doppler_frequency)

    time_interval = 1 / doppler_frequency / channel_oversampling_rate
    timestamps = np.arange(samples_per_drop) * time_interval

    autocorr = np.zeros(2 * samples_per_drop - 1, dtype=complex)

    for drop in range(max_number_of_drops):
        channel.init_drop()

        channel_gains = np.squeeze(
            channel_rayleigh.get_impulse_response(timestamps))
        autocorr += np.correlate(channel_gains,
                                 channel_gains,
                                 'full') / samples_per_drop

    autocorr = autocorr / max_number_of_drops

    # plot distribution of real part
    dt = np.arange(-samples_per_drop + 1, samples_per_drop) * time_interval
    plt.figure()
    plt.subplot(121)
    plt.plot(dt * doppler_frequency, autocorr, label='simulation')
    plt.plot(
        dt *
        doppler_frequency,
        special.j0(
            2 *
            np.pi *
            doppler_frequency *
            dt),
        label='theory')
    plt.xlabel('time (doppler x delay)')
    plt.title('Rayleigh Fading - autocorrelation')
    plt.xlim(-5, 5)
    plt.legend()

    plt.subplot(122)
    doppler_spectrum = np.fft.fftshift(np.fft.fft(autocorr))
    freq = np.fft.fftshift(
        np.fft.fftfreq(
            autocorr.size,
            time_interval)) / doppler_frequency
    plt.plot(freq, np.abs(doppler_spectrum), label='simulation')
    plt.title('Doppler Spectrum')
    plt.xlabel('normalized frequency (f/fd)')
    plt.ylabel('power spectrum density')
    plt.xlim(-doppler_frequency * 1.5, doppler_frequency * 1.5)

    plt.show()

    unittest.main()
