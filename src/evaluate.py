import time

import numpy as np
import torch

from .codec import encode
from .utils import bytes_to_bits

K = 223  # encoded symbols


def evaluate_decoder(decoder, channel_fn, num_samples=1000):
    """Evaluate decoder computing FER and BER in a single pass"""
    results = compare_decoders({"decoder": decoder}, channel_fn, num_samples)
    return results["decoder"]


def compare_decoders(decoders: dict, channel_fn, num_samples=1000):
    """Compare multiple decoders on the same noisy data."""

    metrics = {key: {"fer": 0, "ber": 0} for key in decoders}

    for _ in range(num_samples):
        msg = np.random.bytes(K)
        codeword = encode(msg)
        noisy, erase_pos, _ = channel_fn(codeword)

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


def compute_metrics(predicted_pos, true_pos):
    """Compute precision and recall of predicted error positions."""
    pred = set(predicted_pos)
    true = set(true_pos)

    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0

    return {"precision": precision, "recall": recall}


def benchmark_time(decoders, channel_fn, num_samples=500, warmup=50):
    """Benchmark decoding time per frame for each decoder."""
    test_data = []
    for _ in range(num_samples + warmup):
        msg = np.random.bytes(K)
        codeword = encode(msg)
        noisy, erase_pos, _ = channel_fn(codeword)
        test_data.append((noisy, erase_pos, codeword))

    results = {}
    for key, decoder in decoders.items():
        for noisy, erase_pos, codeword in test_data[:warmup]:
            decoder.decode(noisy, original=codeword, erase_pos=erase_pos)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        for noisy, erase_pos, codeword in test_data[warmup:]:
            decoder.decode(noisy, original=codeword, erase_pos=erase_pos)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start

        results[key] = {
            "total_sec": round(elapsed, 4),
            "per_frame_ms": round(elapsed / num_samples * 1000, 4),
        }

    return results
