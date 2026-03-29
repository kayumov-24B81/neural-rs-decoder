import os

import numpy as np

from codec import encode
from utils import bytes_to_bits

K = 223  # encoded symbols


def evaluate_decoder(decoder, channel_fn, num_samples=1000):
    """Evaluate decoder computing FER and BER in a single pass"""
    results = compare_decoders({"decoder": decoder}, channel_fn, num_samples)
    return results["decoder"]


def compare_decoders(decoders: dict, channel_fn, num_samples=1000):
    """Compare multiple decoders on the same noisy data."""

    metrics = {key: {"fer": 0, "ber": 0} for key in decoders}

    for _ in range(num_samples):
        msg = os.urandom(K)
        codeword = encode(msg)
        noisy, erase_pos = channel_fn(codeword)

        for key in decoders:
            decoded = decoders[key].decode(noisy=noisy, original=codeword, erase_pos=erase_pos)

            if decoded is None or decoded != msg:
                metrics[key]["fer"] += 1

            if decoded is not None:
                msg_bits = bytes_to_bits(msg)
                decoded_bits = bytes_to_bits(decoded)
                metrics[key]["ber"] += np.sum(msg_bits != decoded_bits)
            else:
                metrics[key]["ber"] += K * 8

    return {
        key: {
            "fer": metrics[key]["fer"] / num_samples,
            "ber": metrics[key]["ber"] / (num_samples * K * 8),
        }
        for key in metrics
    }
