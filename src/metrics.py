from dataclasses import dataclass
from typing import Optional

from .codec import NSYM, K

# BM decoder can correct up to 2t = NSYM erasures (here: 32)
# Predicted masks exceeding this bound will be truncated implicitly when
# passed to BM; we track the "overflow" rate as diagnostic.
ERASURE_BUDGET = NSYM


@dataclass
class DecodeResult:
    """ "Result of decoding one block by one decoder, everything a metrics
    accumulator might need, in one place."""

    decoded: Optional[bytes]
    original: bytes
    true_errors: set
    predicted_positions: Optional[set] = None


def init_stats(decoder_names):
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


def _count_bit_errors(decoded, original):
    """Hamming distance in bits between two byte sequences of equal length."""
    return sum(bin(a ^ b).count("1") for a, b in zip(decoded, original))


def update_stats(stats, decoder_name, result):
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


def _safe_div(num, den):
    return num / den if den else float("nan")


def finalize_stats(stats, num_samples):
    """Convert raw counters into final metrics."""
    out = {}
    for name, s in stats.items():
        metrics = {
            "fer": s["fer_num"] / num_samples,
            "dfr": s["dfr_num"] / num_samples,
            "ber": _safe_div(s["ber_num"], s["ber_den"]),
        }
        if s["has_mask"]:
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            metrics["precision"] = _safe_div(tp, tp + fp)
            metrics["recall"] = _safe_div(tp, tp + fn)
            metrics["mask_covers_all"] = s["mask_covers_all_num"] / num_samples
            metrics["num_erasures_mean"] = s["num_erasures_sum"] / num_samples
            metrics["num_erasures_max"] = s["num_erasures_max"]
            metrics["overflow_rate"] = s["overflow_num"] / num_samples
        else:
            metrics["precision"] = float("nan")
            metrics["recall"] = float("nan")
            metrics["mask_covers_all"] = float("nan")
            metrics["num_erasures_mean"] = float("nan")
            metrics["num_erasures_max"] = float("nan")
            metrics["overflow_rate"] = float("nan")
        out[name] = metrics
    return out
