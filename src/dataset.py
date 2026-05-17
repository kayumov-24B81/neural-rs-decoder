import numpy as np
from reedsolo import RSCodec
from torch.utils.data import Dataset

from src.utils import build_input


class RSPositionDataset(Dataset):
    def __init__(
        self, size: int, channel_fns: list, nsym: int = 32, msg_len: int = 223, fixed: bool = False
    ):
        """Generates syncthetic RS transmission data on-the-fly.

        If channel_fns contains multiple callables, each samples picks one
        uniformly at random (mixed channel training)."""

        self.size: int = size
        self.channel_fns: list = channel_fns
        self.nsym: int = nsym
        self.msg_len: int = msg_len
        self.n: int = msg_len + nsym

        self.rsc = RSCodec(nsym)

        self._cache = None
        if fixed:
            self._cache = [self._generate() for _ in range(size)]

    def _generate(self):
        msg = np.random.bytes(self.msg_len)
        codeword = bytes(self.rsc.encode(msg))
        channel_fn = self.channel_fns[np.random.randint(len(self.channel_fns))]
        noisy, _, _ = channel_fn(codeword)

        input_vector = build_input(noisy, self.nsym).astype(np.float32)
        error_pattern = bytes(a ^ b for a, b in zip(codeword, noisy))
        positions_vector = np.array(
            [1.0 if e != 0 else 0.0 for e in error_pattern], dtype=np.float32
        )
        return input_vector, positions_vector, noisy, codeword

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int = None):
        if self._cache is not None:
            input_vec, pos_vec, _, _ = self._cache[idx]
            return input_vec, pos_vec
        input_vec, pos_vec, _, _ = self._generate()
        return input_vec, pos_vec

    def get_raw(self, idx: int):
        if self._cache is None:
            raise RuntimeError("get_raw() requires fixed=True")
        return self._cache[idx]
