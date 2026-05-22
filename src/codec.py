"""RS(255, 223) codec and decoders: classic BM, oracle, and neural-hybrid."""

import numpy as np
import torch
from reedsolo import ReedSolomonError, RSCodec

from .utils import build_input

N = 255
K = 223
NSYM = N - K  # 32 parity symbols
T = NSYM // 2  # 16 — error correction capability


def encode(message: bytes) -> bytearray:
    """Encode a 223-byte message into a 255-byte RS codeword."""
    return bytearray(RSCodec(NSYM).encode(message))


class ClassicDecoder:
    """Standard Berlekamp-Massey RS(255,223) decoder."""

    def __init__(self, nsym: int = NSYM) -> None:
        self.nsym = nsym
        self.rsc = RSCodec(nsym)

    def decode(self, noisy: bytes, erase_pos: int = None, **kwargs) -> bytes | None:
        """Decode received word, return decoded message or None on failure."""
        try:
            decoded, _, _ = self.rsc.decode(noisy, erase_pos=erase_pos)
            return bytes(decoded)
        except ReedSolomonError:
            return None


class OracleDecoder:
    """Ideal decoder that knows true error positions (upper bound on performance)."""

    def __init__(self, nsym: int = NSYM) -> None:
        self.nsym = nsym
        self.decoder = ClassicDecoder(nsym)

    def decode(self, noisy: bytes, original: bytes = None, **kwargs) -> bytes | None:
        """Decode using true error positions as erasures."""
        noisy_arr = bytearray(noisy)
        orig_arr = bytearray(original)
        erase_pos = [i for i in range(len(noisy_arr)) if noisy_arr[i] != orig_arr[i]]
        return self.decoder.decode(noisy, erase_pos=erase_pos)


class HybridDecoder:
    """RS decoder augmented with neural network error position prediction."""

    def __init__(
        self, model: torch.nn.Module, threshold: float = 0.3, nsym: int = NSYM, device: str = "cpu"
    ) -> None:
        """Wrap a positions-predicting mode around a classic RS decoder.

        The model predicts error positions; positions scoring above threshold
        are passed to the RS decoder as erasures.
        """
        self.model = model
        self.threshold = threshold
        self.nsym = nsym
        self.device = device
        self.decoder = ClassicDecoder(nsym)

        self.model.eval()
        self.model.to(device)

    def predict_positions(self, features: np.ndarray) -> list:
        """Predict error positions from precomputed input features."""
        x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)
        positions = (probs[0] > self.threshold).cpu().numpy()
        return [i for i, v in enumerate(positions) if v]

    def decode(
        self, noisy: bytes, features: np.ndarray = None, predicted_positions: list = None, **kwargs
    ) -> bytes | None:
        """Decode using neural-predicted erasure positions, return None on failure."""
        if predicted_positions is None:
            if features is None:
                features = build_input(noisy, self.nsym)
            predicted_positions = self.predict_positions(features)
        return self.decoder.decode(noisy, erase_pos=list(predicted_positions))
