"""Metrics accumulation for decoder benchmarking: FER, BER, mask quality."""

import math
from dataclasses import dataclass
from typing import Optional

from .codec import NSYM, K

# BM corrects errors and erasures under the bound 2*errors + erasures <= NSYM.
# A predicted mask larger than NSYM erasures cannot be decoded at all: reedsolo
# raises ReedSolomonError, ClassicDecoder returns None, and the block counts as
# a decoder failure. We track the "overflow" rate as a diagnostic for how often
# the neural mask blows the erasure budget.
ERASURE_BUDGET = NSYM


@dataclass
class DecodeResult:
    """Result of decoding one block by one decoder.

    Holds everything a metrics accumulator needs for a single block, in one
    place: the decoded output, the ground truth, the true error positions,
    and (for mask-producting decoders) the predicted postions.
    """

    decoded: Optional[bytes]
    original: bytes
    true_errors: set
    predicted_positions: Optional[set] = None


def wilson_interval(count: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score confidence interval for a binomial proportion.

    Returns (low, high) for the given success count out of n trials.
    Default z=1.96 corresponds to a 95% confidence level. Correct near
    p=0 and p=1, unlike the normal approximation; returns (nan, nan)
    for n == 0."""

    if n == 0:
        return (float("nan"), float("nan"))
    p = count / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - margin), min(1.0, center + margin))


def init_stats(decoder_names) -> dict:
    """Initialize a raw-counter dict for a set of decoders."""
    return {
        name: {
            # Common to all decoders
            "fer_num": 0,  # frame errors
            "dfr_num": 0,  # decoder declared failure (returned None)
            "ber_num": 0,  # bit errors
            "ber_den": 0,  # total bits compared
            # Mask-specific (hybrid only)
            "has_mask": False,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "mask_covers_all_num": 0,  # block where mask covers all true_errors
            "num_erasures_sum": 0,  # sum of predicted_positions
            "num_erasures_max": 0,  # max of predicted_positions
            "overflow_num": 0,  # blocks where num of predicted positions > ERASURE_BUDGET
        }
        for name in decoder_names
    }


def _count_bit_errors(decoded: bytes, original: bytes) -> int:
    """Hamming distance in bits between two byte sequences of equal length."""
    return sum(bin(a ^ b).count("1") for a, b in zip(decoded, original))


def update_stats(stats: dict, decoder_name: str, result: DecodeResult) -> None:
    """Update running counters for one decoder with one decode results."""
    s = stats[decoder_name]
    decoded = result.decoded
    original = result.original

    # FER, DFR, BER: common to all decoders
    if decoded is None:
        s["dfr_num"] += 1
        s["fer_num"] += 1
        s["ber_num"] += K * 8  # lost frame = all bits wrong
    else:
        assert len(decoded) == K, f"decoded length {len(decoded)} != K={K}"
        if decoded != original:
            s["fer_num"] += 1
        s["ber_num"] += _count_bit_errors(decoded, original)
    s["ber_den"] += K * 8

    # Mask metrics: only for decoders that produce a mask
    if result.predicted_positions is None:
        return

    s["has_mask"] = True
    predicted = result.predicted_positions
    true_errors = result.true_errors

    tp = len(predicted & true_errors)
    fp = len(predicted - true_errors)
    fn = len(true_errors - predicted)

    assert tp <= len(predicted)
    assert tp <= len(true_errors)

    s["tp"] += tp
    s["fp"] += fp
    s["fn"] += fn

    if true_errors.issubset(predicted):
        s["mask_covers_all_num"] += 1

    n_pred = len(predicted)
    s["num_erasures_sum"] += n_pred
    s["num_erasures_max"] = max(s["num_erasures_max"], n_pred)
    if n_pred > ERASURE_BUDGET:
        s["overflow_num"] += 1


def _safe_div(num, den) -> float:
    """Divide num by den, return NaN instead of raising on den == 0."""
    return num / den if den else float("nan")


def finalize_stats(stats: dict, num_samples: int) -> dict:
    """Convert raw counters into final metrics."""
    out = {}
    for name, s in stats.items():
        fer_lo, fer_hi = wilson_interval(s["fer_num"], num_samples)
        dfr_lo, dfr_hi = wilson_interval(s["dfr_num"], num_samples)
        metrics = {
            "fer": s["fer_num"] / num_samples,
            "fer_ci_low": fer_lo,
            "fer_ci_high": fer_hi,
            "dfr": s["dfr_num"] / num_samples,
            "dfr_ci_low": dfr_lo,
            "dfr_ci_high": dfr_hi,
            "ber": _safe_div(s["ber_num"], s["ber_den"]),
        }
        if s["has_mask"]:
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            metrics["precision"] = _safe_div(tp, tp + fp)
            metrics["recall"] = _safe_div(tp, tp + fn)
            mca_lo, mca_hi = wilson_interval(s["mask_covers_all_num"], num_samples)
            ovf_lo, ovf_hi = wilson_interval(s["overflow_num"], num_samples)
            metrics["mask_covers_all"] = s["mask_covers_all_num"] / num_samples
            metrics["mask_covers_all_ci_low"] = mca_lo
            metrics["mask_covers_all_ci_high"] = mca_hi
            metrics["num_erasures_mean"] = s["num_erasures_sum"] / num_samples
            metrics["num_erasures_max"] = s["num_erasures_max"]
            metrics["overflow_rate"] = s["overflow_num"] / num_samples
            metrics["overflow_rate_ci_low"] = ovf_lo
            metrics["overflow_rate_ci_high"] = ovf_hi
        else:
            metrics["precision"] = float("nan")
            metrics["recall"] = float("nan")
            metrics["mask_covers_all"] = float("nan")
            metrics["mask_covers_all_ci_low"] = float("nan")
            metrics["mask_covers_all_ci_high"] = float("nan")
            metrics["num_erasures_mean"] = float("nan")
            metrics["num_erasures_max"] = float("nan")
            metrics["overflow_rate"] = float("nan")
            metrics["overflow_rate_ci_low"] = float("nan")
            metrics["overflow_rate_ci_high"] = float("nan")
        out[name] = metrics
    return out
