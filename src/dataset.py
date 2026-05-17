import numpy as np
from reedsolo import RSCodec
from torch.utils.data import Dataset

from src.utils import build_input


class RSPositionDataset(Dataset):
    def __init__(self, size: int, channel_fns: list, nsym: int = 32, msg_len: int = 223):
        """Generates syncthetic RS transmission data on-the-fly.

        If channel_fns contains multiple callables, each samples picks one
        uniformly at random (mixed channel training)."""

        self.size: int = size
        self.channel_fns: list = channel_fns
        self.nsym: int = nsym
        self.msg_len: int = msg_len
        self.n: int = msg_len + nsym

        self.rsc = RSCodec(nsym)

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int = None):
        msg: bytes = np.random.bytes(self.msg_len)
        codeword: bytes = self.rsc.encode(msg)
        channel_fn = self.channel_fns[np.random.randint(len(self.channel_fns))]
        noisy, _, _ = channel_fn(codeword)

        input_vector = build_input(noisy, self.nsym).astype(np.float32)

        error_pattern = bytes(a ^ b for a, b in zip(codeword, noisy))
        positions_vector = np.array(
            [1.0 if e != 0 else 0.0 for e in error_pattern], dtype=np.float32
        )

        return input_vector, positions_vector
