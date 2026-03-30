import os

import numpy as np
from reedsolo import RSCodec, rs_calc_syndromes
from torch.utils.data import Dataset

from .channel import qsc_erasure_channel
from .utils import bytes_to_bits, get_zero_mask


class RSPositionDataset(Dataset):
    def __init__(self, size: int, p_err: float, p_erase: float, nsym: int = 32, msg_len: int = 223):
        """Initializes the dataset and generates samples"""
        self.size: int = size
        self.p_err: float = p_err
        self.p_erase: float = p_erase
        self.nsym: int = nsym
        self.msg_len: int = msg_len
        self.n: int = msg_len + nsym

        self.rsc = RSCodec(nsym)

        self.inputs: np.ndarray
        self.positions: np.ndarray

        self._generate_data()

    def _generate_data(self):
        """Generate syntheric RS transmission data using QSC erasure/error channel"""
        inputs = []
        positions = []

        for _ in range(self.size):
            msg: bytes = os.urandom(self.msg_len)
            codeword: bytes = self.rsc.encode(msg)
            noisy, _, _ = qsc_erasure_channel(codeword, self.p_err, self.p_erase)

            syndrome = rs_calc_syndromes(noisy, self.nsym)[1:]
            syndrome_bits = bytes_to_bits(bytes(syndrome))
            zero_mask = get_zero_mask(noisy)
            input_vector = np.concatenate([syndrome_bits, zero_mask])

            error_pattern = bytes(a ^ b for a, b in zip(codeword, noisy))
            positions_vector = np.array(
                [1.0 if e != 0 else 0.0 for e in error_pattern], dtype=np.float32
            )

            inputs.append(input_vector)
            positions.append(positions_vector)

        self.inputs = np.array(inputs, dtype=np.float32)
        self.positions = np.array(positions, dtype=np.float32)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.size

    def __getitem__(self, idx: int):
        """Retrieves a single dataset sample."""
        return self.inputs[idx], self.positions[idx]
