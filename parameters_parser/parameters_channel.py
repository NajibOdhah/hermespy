import configparser
import os

import numpy as np

from parameters_parser.parameters_repetition_encoder import ParametersRepetitionEncoder


class ParametersChannel:
    """This class implements the parser of the channel model parameters.

    Attributes:
        multipath_model_val list(str): list of valid channel multipath models.
        supported_encoders List(str): list of valid encoders
    """

    multipath_model_val = [
        "NONE",
        "STOCHASTIC",
        "COST259",
        "EXPONENTIAL",
        "QUADRIGA"]
    cost_259_type_val = ["TYPICAL_URBAN", "RURAL_AREA", "HILLY_TERRAIN"]
    supported_encoders = ["REPETITION"]
    # truncate exponential profile for values less than this
    exponential_truncation = 1e-5

    def __init__(self) -> None:
        """Creates a parsing object, that will manage the channel parameters."""
        self.multipath_model = ""
        self.delays = np.array([])
        self.power_delay_profile_db = np.array([])
        self.power_delay_profile = np.array([])
        self.k_factor_rice = np.array([])
        self.velocity = np.array([])
        self.attenuation_db = 0.
        self.gain = 0.

        self.rms_delay_spread = 0.

        # parameters for COST 259 channel model only
        # previous parameters are derived from them
        self.cost_259_type = ""
        # parameter relevant for COST 259 hilly-terrain model only (angle of
        # LOS component)
        self.los_theta_0 = 0.

        # parameters for exponential channel model only
        # previous parameters are derived from them
        self.tap_interval = 0.
        self.rms_delay = 0.

        ###
        # parameters for Quadriga
        self.matlab_or_octave = 0.
        self.scenario_qdg = 0.
        self.antenna_kind = 0.
        self.path_quadriga_src = 0.
        ###

        self.dir_encoding_parameters = os.path.join(
            os.getcwd(), '_settings', 'coding')
        self._encoder_param_file = os.path.join(
            self.dir_encoding_parameters,
            'settings_repetition_encoder.ini')
        self._encoded_bits_n = 1
        self._data_bits_k = 1

    def read_params(self, section: configparser.SectionProxy) -> None:
        """Reads channel parameters of a given config file.

        Args:
            section (configparser.SectionProxy): Section in the file to read the
                parameters from.
        """

        self.multipath_model = section.get(
            "multipath_model", fallback='none').upper()

        if self.multipath_model == "STOCHASTIC":
            self.delays = section.get("delays", fallback='0')
            self.delays = np.fromstring(self.delays, sep=',')

            self.power_delay_profile_db = section.get(
                "power_delay_profile_db", fallback='0')
            self.power_delay_profile_db = np.fromstring(
                self.power_delay_profile_db, sep=',')

            k_factor_rice_db_str = section.get("k_rice_db", fallback='-inf')
            k_factor_rice_db = np.fromstring(k_factor_rice_db_str, sep=',')
            self.k_factor_rice = 10 ** (k_factor_rice_db / 10)

        if self.multipath_model == "COST259":
            self.cost_259_type = section.get(
                "cost_type", fallback='typical_urban').upper()

        if self.multipath_model == "EXPONENTIAL":
            self.tap_interval = section.getfloat("tap_interval")
            self.rms_delay = section.getfloat("rms_delay")

        self.attenuation_db = section.getfloat("attenuation_db", fallback=0)

        # encoding related
        self._encoder_param_file = section.get(
            "encoder_param_file", fallback=self._encoder_param_file)
        self._encoded_bits_n = section.getint("encoded_bits_n", fallback=1)
        self._data_bits_k = section.getint("data_bits_k", fallback=1)

    def check_params(self) -> None:
        """Checks the validity of the read parameters."""

        if self.multipath_model not in ParametersChannel.multipath_model_val:
            raise ValueError(
                "multipath_model '" +
                self.multipath_model +
                "' not supported")

        if self.multipath_model == "STOCHASTIC":
            if self.delays.shape != self.power_delay_profile_db.shape:
                raise ValueError(
                    "'power_delay_profile' must have the same length as 'delays'")

            if self.delays.shape != self.k_factor_rice.shape:
                raise ValueError(
                    "'power_delay_profile' must have the same length as 'delays'")

        elif self.multipath_model == "COST259":
            if self.cost_259_type not in ParametersChannel.cost_259_type_val:
                raise ValueError(
                    'COST 259 type (' + self.cost_259_type + ') not supported')
            elif self.cost_259_type == 'TYPICAL_URBAN':
                self.delays = np.asarray([0, .217, .512, .514, .517, .674, .882, 1.230, 1.287, 1.311, 1.349, 1.533,
                                          1.535, 1.622, 1.818, 1.836, 1.884, 1.943, 2.048, 2.140]) * 1e-6
                self.power_delay_profile_db = np.asarray([-5.7, - 7.6, -10.1, -10.2, -10.2, -11.5, -13.4, -16.3, -16.9,
                                                          -17.1, -17.4, -19.0, -19.0, -19.8, -21.5, -21.6, -22.1, -22.6,
                                                          -23.5, -24.3])
                self.k_factor_rice = np.zeros(self.delays.shape)

            elif self.cost_259_type == 'RURAL_AREA':
                self.delays = np.asarray(
                    [0, .042, .101, .129, .149, .245, .312, .410, .469, .528]) * 1e-6
                self.power_delay_profile_db = np.asarray([-5.2, -6.4, -8.4, -9.3, -10.0, -13.1, -15.3, -18.5, -20.4,
                                                          -22.4])
                self.k_factor_rice = np.zeros(self.delays.shape)

            elif self.cost_259_type == 'HILLY_TERRAIN':
                self.delays = np.asarray([0, .356, .441, .528, .546, .609, .625, .842, .916, .941, 15.0, 16.172, 16.492,
                                          16.876, 16.882, 16.978, 17.615, 17.827, 17.849, 18.016]) * 1e-6
                self.power_delay_profile_db = np.asarray([-3.6, -8.9, -10.2, -11.5, -11.8, -12.7, -13.0, -16.2, -17.3,
                                                          -17.7, -17.6, -22.7, -24.1, -25.8, -25.8, -26.2, -29.0, -29.9,
                                                          -30.0, -30.7])
                self.k_factor_rice = np.hstack(
                    (np.inf, np.zeros(self.delays.size - 1)))
                self.los_theta_0 = np.arccos(.7)

            self.multipath_model = "STOCHASTIC"

        elif self.multipath_model == "EXPONENTIAL":
            # calculate the decay exponent alpha based on an infinite power delay profile, in which case
            # rms_delay = exp(-alpha/2)/(1-exp(-alpha)), cf. geometric distribution
            # Truncate the distributions for paths whose average power is very
            # small (less than exponential_truncation)
            rms_norm = self.rms_delay / self.tap_interval
            alpha = -2 * \
                np.log((-1 + np.sqrt(1 + 4 * rms_norm ** 2)) / (2 * rms_norm))
            max_delay_in_samples = - \
                int(np.ceil(np.log(ParametersChannel.exponential_truncation) / alpha))
            self.delays = np.arange(
                max_delay_in_samples + 1) * self.tap_interval
            self.power_delay_profile_db = 10 * \
                np.log10(np.exp(-alpha * np.arange(max_delay_in_samples + 1)))
            self.k_factor_rice = np.zeros(self.delays.shape)

            self.multipath_model = "STOCHASTIC"

        ####
        elif self.multipath_model == "QUADRIGA":
            # see quadriga_doc
            self.multipath_model = "QUADRIGA"
        ###

        if self.multipath_model != "NONE" and self.multipath_model != "QUADRIGA":
            self.power_delay_profile = 10 ** (self.power_delay_profile_db / 10)
            self.power_delay_profile = self.power_delay_profile / \
                sum(self.power_delay_profile)

        self.gain = 10 ** (-self.attenuation_db / 20)

        # read encoder parameters file
        config = configparser.ConfigParser()
        encoding_params_file_path = os.path.join(
            self.dir_encoding_parameters, self._encoder_param_file)

        if not os.path.exists(encoding_params_file_path):
            raise ValueError(
                f'File {encoding_params_file_path} does not exist.')

        config.read(encoding_params_file_path)
        self.encoding_type = config["General"].get("type").upper()

        if self.encoding_type not in self.supported_encoders:
            raise ValueError(
                f"Encoding Type {self.encoding_type} not supported")

        if self.encoding_type == "REPETITION":
            encoding_parameters = ParametersRepetitionEncoder()

        encoding_parameters.encoded_bits_n = self._encoded_bits_n
        encoding_parameters.data_bits_k = self._data_bits_k

        encoding_parameters.read_params(config["General"])
        self.encoding = encoding_parameters
