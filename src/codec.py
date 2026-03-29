import torch
from reedsolo import ReedSolomonError, RSCodec

from utils import build_input

N = 255
K = 223
NSYM = N - K  # 32 parity symbols
T = NSYM // 2  # 16 — error correction capability

_rsc = RSCodec(NSYM)


def encode(message):
    """Encode a 223-byte message into a 255-byte RS codeword."""
    return bytearray(_rsc.encode(message))


class ClassicDecoder:
    """Standard Berlekamp-Massey RS(255,223) decoder."""

    def __init__(self, nsym=NSYM):
        self.nsym = nsym
        self.rsc = RSCodec(nsym)

    def decode(self, noisy, erase_pos=None, **kwargs):
        """Decode received word, return decoded message or None on failure."""
        try:
            decoded, _, _ = self.rsc.decode(noisy, erase_pos=erase_pos)
            return bytes(decoded)
        except ReedSolomonError:
            return None


class OracleDecoder:
    """Ideal decoder that knows true error positions (upper bound on performance)."""

    def __init__(self, nsym=NSYM):
        self.nsym = nsym
        self.decoder = ClassicDecoder(nsym)

    def decode(self, noisy, original=None, **kwargs):
        """Decode using true error positions as erasures."""
        noisy_arr = bytearray(noisy)
        orig_arr = bytearray(original)
        erase_pos = [i for i in range(len(noisy_arr)) if noisy_arr[i] != orig_arr[i]]
        return self.decoder.decode(noisy, erase_pos=erase_pos)


class HybridDecoder:
    """RS decoder augmented with neural network error position prediction."""

    def __init__(self, model, threshold=0.3, nsym=NSYM, device="cpu"):
        self.model = model
        self.threshold = threshold
        self.nsym = nsym
        self.device = device
        self.decoder = ClassicDecoder(nsym)

        self.model.eval()
        self.model.to(device)

    def predict_positions(self, features):
        """Predict error positions from precomputed input features."""
        x = torch.tensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)
        positions = (probs[0] > self.threshold).cpu().numpy()
        return [i for i, v in enumerate(positions) if v]

    def decode(self, noisy, features=None, erase_pos=None, **kwargs):
        """Decode using neural-predicted erasure positions, return None on failure."""
        if features is None:
            features = build_input(noisy, self.nsym)
        erase_pos = self.predict_positions(features)
        return self.decoder.decode(noisy, erase_pos=erase_pos)
