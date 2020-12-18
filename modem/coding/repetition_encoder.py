from typing import List

import numpy as np

from modem.coding.encoder import Encoder
from parameters_parser.parameters_repetition_encoder import ParametersRepetitionEncoder


class RepetitionEncoder(Encoder):
    """Exemplary implementation of a RepetitionEncoder.

    This shows how a new encoder is to implemented. It's a three-step process:

    1. override the `encode` function
    2. derive a parameters parser from the super class ParametersEncoder
    3. implement a deoder
    """

    def __init__(self, params: ParametersRepetitionEncoder,
                 bits_in_frame: int) -> None:
        self.params = params
        self.params.data_bits_k = 1
        self.bits_in_frame = bits_in_frame

    def encode(self, data_bits: List[np.array]) -> List[np.array]:
        if self.encoded_bits_n > 1:
            block_size = int(np.floor(self.bits_in_frame / len(data_bits)))
            for block_idx, block in enumerate(data_bits):
                block = np.repeat(block, self.encoded_bits_n)
                block = np.append(
                    block,
                    np.zeros(block_size - len(block))
                )
                data_bits[block_idx] = block
        return data_bits


class RepetitionDecoder:
    """Exemplary implementation of a Repetition Decoder.

    It must provide the `decode` method.
    """

    def __init__(self, encoder: RepetitionEncoder) -> None:
        self.encoder = encoder

    def decode(self, encoded_bits: List[np.array]) -> List[np.array]:
        if self.encoder.encoded_bits_n > 1:
            decoded_bits = [np.array([]) for block in encoded_bits]
            for block_idx, block in enumerate(encoded_bits):
                no_data_bits_in_block = int(
                    len(block) / self.encoder.encoded_bits_n)
                initial_index = 0

                for decoded_bits_idx in range(0, no_data_bits_in_block):
                    code_block = block[initial_index:initial_index +
                                       self.encoder.encoded_bits_n]
                    decoded_bits[block_idx] = np.append(
                        decoded_bits[block_idx],
                        self._majority_vote(code_block)
                    )
                    initial_index += self.encoder.encoded_bits_n
        else:
            decoded_bits = encoded_bits

        return decoded_bits

    def _majority_vote(self, encoded_bits: np.array) -> int:
        if np.mean(encoded_bits) > 0.5:
            return 1
        else:
            return 0
