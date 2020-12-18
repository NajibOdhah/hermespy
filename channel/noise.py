import numpy as np


class Noise:
    """Implements a complex additive white gaussian noise at the receiver.

    Attributes:
        snr_type (str)
            ignal-to-noise definition.
            The following values are currently supported:
             - EB/N0(dB): bit-energy over N0 (in dB)
             - ES/NO(dB): signal-energy over N0 (in dB)
             - custom(dB): custom SNR (in dB)
    """

    def __init__(self, snr_type: str, rnd: np.random.RandomState) -> None:
        self.snr_type = snr_type

        self._rnd = rnd

    def add_noise(self, signal: np.array, snr: float,
                  signal_energy: float) -> np.ndarray:
        """Adds noise to a signal.

        The SNR definition is defined in self.snr_type.

        Args:
            signal (np.array): Input signal.
            snr (float):
                Signal-to-noise ratio for noise power. If snr_type is 'CUSTOM',
                the parameter is interpreted as actual noise variance.
            signal_energy: Energy of signal.

        Returns:
            np.ndarray: Noisy signal.
        """
        if self.snr_type == 'EB/N0(DB)' or self.snr_type == 'ES/N0(DB)':
            snr_linear = self.db_to_linear(snr)
            noise_var = signal_energy / snr_linear
        elif self.snr_type.upper() == 'CUSTOM':
            noise_var = self.db_to_linear(snr)
        else:
            raise ValueError("Invalid snr_type")

        noise = (self._rnd.standard_normal(signal.shape) + 1j *
                 self._rnd.standard_normal(signal.shape)) / np.sqrt(2) * np.sqrt(noise_var)
        return signal + noise

    def db_to_linear(self, db_quantity: float) -> float:
        """Converts db quantity to linear quanity."""
        return (10**(db_quantity / 10.))
