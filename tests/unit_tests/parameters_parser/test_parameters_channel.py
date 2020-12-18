import unittest

import numpy as np

from parameters_parser.parameters_channel import ParametersChannel


class TestParametersChannel(unittest.TestCase):
    def setUp(self) -> None:
        self.params_channel = ParametersChannel()

    def test_check_params_proper_multipath_model_val(self) -> None:
        self.params_channel.multipath_model = "INVALID_PATH_MODEL"
        self.assertRaises(
            ValueError,
            lambda: self.params_channel.check_params())

    def test_check_params_stochastic_path_model(self) -> None:
        self.params_channel.multipath_model = "STOCHASTIC"
        self.params_channel.power_delay_profile_db = np.fromstring(
            "3,5", sep=",")
        self.params_channel.delays = np.fromstring("3", sep=",")
        self.assertRaises(
            ValueError,
            lambda: self.params_channel.check_params())

        self.params_channel.delays = np.fromstring("3,5", sep=",")
        self.params_channel.k_factor_rice = np.fromstring("3", sep=",")
        self.assertRaises(
            ValueError,
            lambda: self.params_channel.check_params())

    def test_check_params_cost259_path_model_typical_urban(self) -> None:
        self.params_channel.multipath_model = "COST259"
        self.params_channel.cost_259_type = "INVALID_COST_TYPE"
        self.assertRaises(
            ValueError,
            lambda: self.params_channel.check_params())

        self.params_channel.cost_259_type = "TYPICAL_URBAN"
        self.params_channel.check_params()
        self.assertEqual(self.params_channel.multipath_model, "STOCHASTIC")

        np.testing.assert_array_almost_equal(
            self.params_channel.delays,
            np.array([0, 0.217, 0.512, 0.514, 0.517, 0.674, 0.882, 1.230,
                      1.287, 1.311, 1.349, 1.533, 1.535, 1.622, 1.818,
                      1.836, 1.884, 1.943, 2.048, 2.140])
            * 1e-6,
        )

        np.testing.assert_array_almost_equal(self.params_channel.power_delay_profile_db,
                                             np.asarray([-5.7, - 7.6, -10.1, -10.2, -10.2, -11.5, -13.4, -16.3, -16.9,
                                                         -17.1, -17.4, -19.0, -19.0, -19.8, -21.5, -21.6, -22.1, -22.6,
                                                         -23.5, -24.3]))

        self.assertEqual(
            self.params_channel.k_factor_rice.shape,
            self.params_channel.delays.shape)

    def test_check_params_cost259_path_model_rural_area(self) -> None:
        self.params_channel.multipath_model = 'COST259'
        self.params_channel.cost_259_type = 'RURAL_AREA'

        self.params_channel.check_params()

        np.testing.assert_array_almost_equal(self.params_channel.delays,
                                             np.array([0, .042, .101, .129, .149, .245, .312, .410, .469, .528]) * 1e-6)

        np.testing.assert_array_almost_equal(self.params_channel.power_delay_profile_db,
                                             np.array([-5.2, -6.4, -8.4, -9.3, -10.0, -13.1, -15.3, -18.5, -20.4, -22.4]))

        self.assertEqual(
            self.params_channel.k_factor_rice.shape,
            self.params_channel.delays.shape)

    def test_check_params_cost259_path_model_hilly_terrain(self) -> None:
        self.params_channel.multipath_model = 'COST259'
        self.params_channel.cost_259_type = 'HILLY_TERRAIN'

        self.params_channel.check_params()

        np.testing.assert_array_almost_equal(self.params_channel.delays,
                                             np.array([0, .356, .441, .528, .546, .609, .625, .842, .916, .941, 15.0, 16.172, 16.492,
                                                       16.876, 16.882, 16.978, 17.615, 17.827, 17.849, 18.016]) * 1e-6)

        np.testing.assert_array_almost_equal(self.params_channel.power_delay_profile_db,
                                             np.array([-3.6, -8.9, -10.2, -11.5, -11.8, -12.7, -13.0, -16.2, -17.3,
                                                       -17.7, -17.6, -22.7, -24.1, -25.8, -25.8, -26.2, -29.0, -29.9,
                                                       -30.0, -30.7]))

        k_factor_rice_expected = np.hstack(
            (np.inf, np.zeros(self.params_channel.delays.size - 1))
        )
        np.testing.assert_array_almost_equal(
            self.params_channel.k_factor_rice,
            k_factor_rice_expected)
        self.assertEqual(self.params_channel.los_theta_0, np.arccos(0.7))

    def test_check_params_exponential(self) -> None:
        rms_delay = 3e-6
        tap_interval = 1e-6

        self.params_channel.multipath_model = 'EXPONENTIAL'
        self.params_channel.rms_delay = rms_delay
        self.params_channel.tap_interval = tap_interval

        rms_norm = rms_delay / tap_interval
        alpha = -2 * \
            np.log((-1 + np.sqrt(1 + 4 * rms_norm ** 2)) / (2 * rms_norm))
        max_delay_in_samples = - \
            int(np.ceil(np.log(ParametersChannel.exponential_truncation) / alpha))

        delays = np.arange(max_delay_in_samples + 1) * tap_interval
        power_delay_profile_db = 10 * \
            np.log10(np.exp(-alpha * np.arange(max_delay_in_samples + 1)))
        k_factor_rice = np.zeros(delays.shape)

        self.params_channel.check_params()
        np.testing.assert_array_almost_equal(
            delays, self.params_channel.delays)
        np.testing.assert_array_almost_equal(
            power_delay_profile_db,
            self.params_channel.power_delay_profile_db)
        self.assertEqual(k_factor_rice.shape,
                         self.params_channel.k_factor_rice.shape)
        self.assertEqual(self.params_channel.multipath_model, 'STOCHASTIC')

    def test_check_params_no_multipath_model_provided(self) -> None:
        self.params_channel.multipath_model = 'NONE'
        self.assertEqual(self.params_channel.power_delay_profile.size, 0)

    def test_check_params_power_delay_profile_calculation(self) -> None:
        power_delay_profile_db = np.array([1, 2, 3])

        self.params_channel.multipath_model = 'STOCHASTIC'
        self.params_channel.power_delay_profile_db = power_delay_profile_db
        self.params_channel.delays = np.zeros(power_delay_profile_db.shape)
        self.params_channel.k_factor_rice = np.zeros(
            power_delay_profile_db.shape)

        self.params_channel.check_params()

        power_delay_profile = 10 ** (power_delay_profile_db / 10)
        power_delay_profile = power_delay_profile / sum(power_delay_profile)

        np.testing.assert_array_almost_equal(
            self.params_channel.power_delay_profile,
            power_delay_profile)
