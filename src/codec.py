import numpy as np
import torch
from reedsolo import ReedSolomonError, RSCodec

from utils import compute_syndrome_bits, get_zero_mask


class HybridDecoder:
    def __init__(self, model, threshold=0.3, nsym=32, device="cpu"):
        self.model = model
        self.threshold = threshold
        self.nsym = nsym
        self.device = device
        self.rsc = RSCodec(nsym)

        self.model.eval()
        self.model.to(device)

    def predict_positions(self, noisy):
        syndrome_bits = compute_syndrome_bits(noisy, self.nsym)
        zero_mask = get_zero_mask(noisy)
        inp = np.concatenate([syndrome_bits, zero_mask])

        x = torch.tensor(inp).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)

        positions = (probs[0] > self.threshold).cpu().numpy()
        return [i for i, v in enumerate(positions) if v]

    def decode(self, noisy):
        erase_pos = self.predict_positions(noisy)

        try:
            decoded, _, _ = self.rsc.decode(noisy, erase_pos=erase_pos)
            return bytes(decoded)
        except ReedSolomonError:
            return None


class ClassicDecoder:
    def __init__(self, nsym=32):
        self.nsym = nsym
        self.rsc = RSCodec(nsym)

    def decode(self, noisy, erase_pos=None):
        try:
            decoded, _, _ = self.rsc.decode(noisy, erase_pos=erase_pos)
            return bytes(decoded)
        except ReedSolomonError:
            return None

    def encode(self, message):
        return self.rsc.encode(message)
