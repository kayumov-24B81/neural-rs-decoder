import math

import pytest

from src.codec import K
from src.metrics import ERASURE_BUDGET, DecodeResult, finalize_stats, init_stats, update_stats

# HELPERS


def make_msg(byte_value=0):
    """K-byte message of a repeated value (predictable BER)."""
    return bytes([byte_value]) * K


def make_result(decoded, original, true_errors, predicted_positions=None):
    """Build a DecodeResult with explicit fields."""
    return DecodeResult(
        decoded=decoded,
        original=original,
        true_errors=set(true_errors),
        predicted_positions=set(predicted_positions) if predicted_positions is not None else None,
    )


# init_stats


def test_init_stats_creates_entry_per_decoder():
    stats = init_stats(["classic", "oracle", "neural"])
    assert set(stats.keys()) == {"classic", "oracle", "neural"}
    for s in stats.values():
        assert s["fer_num"] == 0
        assert s["dfr_num"] == 0
        assert s["ber_num"] == 0
        assert s["ber_den"] == 0


def test_init_stats_has_mask_starts_false():
    stats = init_stats(["any"])
    assert stats["any"]["has_mask"] is False


# update_stats: common metrics


def test_update_perfect_decode():
    """decoded == original -> no FER, no DFR, no BER."""
    stats = init_stats(["d"])
    msg = make_msg()
    result = make_result(decoded=msg, original=msg, true_errors=set())
    update_stats(stats, "d", result)

    assert stats["d"]["fer_num"] == 0
    assert stats["d"]["dfr_num"] == 0
    assert stats["d"]["ber_num"] == 0
    assert stats["d"]["ber_den"] == K * 8


def test_update_decoder_failure_counts_as_dfr_and_fer():
    """decoded is None -> DFR++, FER++, BER += K*8"""
    stats = init_stats(["d"])
    msg = make_msg()
    result = make_result(decoded=None, original=msg, true_errors={1, 2, 3})
    update_stats(stats, "d", result)

    assert stats["d"]["dfr_num"] == 1
    assert stats["d"]["fer_num"] == 1
    assert stats["d"]["ber_num"] == K * 8
    assert stats["d"]["ber_den"] == K * 8


def test_update_silent_error_counts_fer_not_dfr():
    """decoded != original but not None -> FER++, DFR unchanged."""
    stats = init_stats(["d"])
    original = make_msg(byte_value=0x00)
    decoded = make_msg(byte_value=0xFF)
    result = make_result(decoded=decoded, original=original, true_errors={5})
    update_stats(stats, "d", result)

    assert stats["d"]["fer_num"] == 1
    assert stats["d"]["dfr_num"] == 0
    assert stats["d"]["ber_num"] == K * 8

    assert stats["d"]["fer_num"] == 1
    assert stats["d"]["dfr_num"] == 0
    assert stats["d"]["ber_num"] == K * 8


def test_update_ber_counts_bit_hamming_distance():
    """BER counter is Hamming distance in bits, not byte mismatches"""
    stats = init_stats(["d"])
    original = bytes([0b00000000]) * K
    decoded = bytes([0b00000001]) * K
    result = make_result(decoded=decoded, original=original, true_errors=set())
    update_stats(stats, "d", result)

    assert stats["d"]["ber_num"] == K


# update_stats: mask metrics


def test_update_no_mask_leaves_counters_untouched():
    """For decoders without mask, tp/fp/fn/etc. stay zero."""
    stats = init_stats(["classic"])
    msg = make_msg()
    result = make_result(decoded=msg, original=msg, true_errors={1, 2}, predicted_positions=None)
    update_stats(stats, "classic", result)

    s = stats["classic"]
    assert s["has_mask"] is False
    assert s["tp"] == 0
    assert s["fp"] == 0
    assert s["fn"] == 0
    assert s["mask_covers_all_num"] == 0
    assert s["num_erasures_sum"] == 0
    assert s["num_erasures_max"] == 0
    assert s["overflow_num"] == 0


def test_update_with_mask_sets_has_mask_true():
    stats = init_stats(["neural"])
    msg = make_msg()
    result = make_result(decoded=msg, original=msg, true_errors={1}, predicted_positions={1})
    update_stats(stats, "neural", result)
    assert stats["neural"]["has_mask"] is True


def test_update_perfect_mask_gives_full_recall_and_precision():
    """predicted == true_errors -> tp = |true|, fp = 0, fn = 0."""
    stats = init_stats(["neural"])
    msg = make_msg()
    true = {3, 7, 11}
    result = make_result(decoded=msg, original=msg, true_errors=true, predicted_positions=true)
    update_stats(stats, "neural", result)

    assert stats["neural"]["tp"] == 3
    assert stats["neural"]["fp"] == 0
    assert stats["neural"]["fn"] == 0


def test_update_empty_mask_gives_full_fn():
    """predicted = empty, true =/ empty -> all errors are false negatives."""
    stats = init_stats(["neural"])
    msg = make_msg()
    true = {3, 7, 11}
    result = make_result(decoded=msg, original=msg, true_errors=true, predicted_positions=set())
    update_stats(stats, "neural", result)

    assert stats["neural"]["tp"] == 0
    assert stats["neural"]["fp"] == 0
    assert stats["neural"]["fn"] == 3


def test_update_mask_covers_all_when_subset():
    """predicted ⊇ true_errors -> mask_covers_all_num++."""
    stats = init_stats(["neural"])
    msg = make_msg()
    true = {1, 2}
    predicted = {1, 2, 3, 4}
    result = make_result(decoded=msg, original=msg, true_errors=true, predicted_positions=predicted)
    update_stats(stats, "neural", result)

    assert stats["neural"]["mask_covers_all_num"] == 1
    assert stats["neural"]["tp"] == 2
    assert stats["neural"]["fp"] == 2
    assert stats["neural"]["fn"] == 0


def test_update_mask_covers_all_fails_on_missed_position():
    """One missed error -> mask_covers_all_num unchanged."""
    stats = init_stats(["neural"])
    msg = make_msg()
    true = {1, 2, 3}
    predicted = {1, 2}
    result = make_result(decoded=msg, original=msg, true_errors=true, predicted_positions=predicted)
    update_stats(stats, "neural", result)

    assert stats["neural"]["mask_covers_all_num"] == 0
    assert stats["neural"]["fn"] == 1


# update_stats: overflow


def test_overflow_below_budget_not_counted():
    stats = init_stats(["neural"])
    msg = make_msg()
    predicted = set(range(ERASURE_BUDGET))
    result = make_result(
        decoded=msg, original=msg, true_errors=set(), predicted_positions=predicted
    )
    update_stats(stats, "neural", result)
    assert stats["neural"]["overflow_num"] == 0


def test_overflow_above_budget_counted():
    stats = init_stats(["neural"])
    msg = make_msg()
    predicted = set(range(ERASURE_BUDGET + 1))
    result = make_result(
        decoded=msg, original=msg, true_errors=set(), predicted_positions=predicted
    )
    update_stats(stats, "neural", result)
    assert stats["neural"]["overflow_num"] == 1


def test_num_erasures_max_tracks_running_max():
    stats = init_stats(["neural"])
    msg = make_msg()
    for size in [3, 7, 5, 10, 4]:
        predicted = set(range(size))
        result = make_result(
            decoded=msg, original=msg, true_errors=set(), predicted_positions=predicted
        )
        update_stats(stats, "neural", result)
    assert stats["neural"]["num_erasures_max"] == 10
    assert stats["neural"]["num_erasures_sum"] == 3 + 7 + 5 + 10 + 4


# finalize_stats


def test_finalize_no_mask_decoder_yields_nan_for_mask_metrics():
    stats = init_stats(["classic"])
    msg = make_msg()
    result = make_result(decoded=msg, original=msg, true_errors=set())
    update_stats(stats, "classic", result)

    metrics = finalize_stats(stats, num_samples=1)
    assert math.isnan(metrics["classic"]["precision"])
    assert math.isnan(metrics["classic"]["recall"])
    assert math.isnan(metrics["classic"]["mask_covers_all"])
    assert math.isnan(metrics["classic"]["num_erasures_mean"])
    assert math.isnan(metrics["classic"]["overflow_rate"])


def test_finalize_normalizes_by_num_samples():
    stats = init_stats(["d"])
    msg = make_msg()

    for _ in range(3):
        update_stats(stats, "d", make_result(decoded=msg, original=msg, true_errors=set()))
    for _ in range(2):
        update_stats(stats, "d", make_result(decoded=None, original=msg, true_errors={1}))

    metrics = finalize_stats(stats, num_samples=5)
    assert metrics["d"]["fer"] == pytest.approx(2 / 5)
    assert metrics["d"]["dfr"] == pytest.approx(2 / 5)


def test_finalize_precision_recall_from_accumulated_tp_fp_fn():
    """precision = tp / (tp + fp), recall = tp / (tp + fn)."""
    stats = init_stats(["neural"])
    msg = make_msg()

    # tp=1, fp=1, fn=1
    update_stats(
        stats,
        "neural",
        make_result(decoded=msg, original=msg, true_errors={1, 3}, predicted_positions={1, 2}),
    )
    # tp=2, fp=0, fn=0
    update_stats(
        stats,
        "neural",
        make_result(decoded=msg, original=msg, true_errors={5, 6}, predicted_positions={5, 6}),
    )
    # should be: tp=3, fp=1, fn=1
    metrics = finalize_stats(stats, num_samples=2)

    assert metrics["neural"]["precision"] == pytest.approx(3 / 4)
    assert metrics["neural"]["recall"] == pytest.approx(3 / 4)


def test_finalize_precision_handles_zero_predictions():
    """tp + fp = 0 -> precision = NaN, no crash."""
    stats = init_stats(["neural"])
    msg = make_msg()

    update_stats(
        stats,
        "neural",
        make_result(decoded=msg, original=msg, true_errors={1}, predicted_positions=set()),
    )
    metrics = finalize_stats(stats, num_samples=1)
    assert math.isnan(metrics["neural"]["precision"])
    assert metrics["neural"]["recall"] == pytest.approx(0.0)


def test_finalize_recall_handle_zero_true_errors():
    """tp + fn = 0 -> recall = Nan, no crash."""
    stats = init_stats(["neural"])
    msg = make_msg()

    update_stats(
        stats,
        "neural",
        make_result(decoded=msg, original=msg, true_errors=set(), predicted_positions={1}),
    )
    metrics = finalize_stats(stats, num_samples=1)
    assert math.isnan(metrics["neural"]["recall"])
    assert metrics["neural"]["precision"] == pytest.approx(0.0)


# integration


def test_full_flow_classic_and_neural_decoders():
    """Multi-decoder run: counters don't cross-contaminate."""
    stats = init_stats(["classic", "neural"])
    msg = make_msg()

    update_stats(
        stats,
        "classic",
        make_result(decoded=msg, original=msg, true_errors={1, 2}),
    )
    update_stats(
        stats,
        "neural",
        make_result(decoded=msg, original=msg, true_errors={1, 2}, predicted_positions={1, 2}),
    )

    update_stats(
        stats,
        "classic",
        make_result(decoded=None, original=msg, true_errors={3, 4, 5}),
    )
    update_stats(
        stats,
        "neural",
        make_result(decoded=msg, original=msg, true_errors={3, 4, 5}, predicted_positions={3, 4}),
    )

    metrics = finalize_stats(stats, num_samples=2)

    # classic: one None, one perfect
    assert metrics["classic"]["fer"] == pytest.approx(0.5)
    assert metrics["classic"]["dfr"] == pytest.approx(0.5)
    assert math.isnan(metrics["classic"]["precision"])

    # neural: all decoded match msg, but mask has errors
    # tp=2+2=4, fp=0, fn=0+1=1

    assert metrics["neural"]["fer"] == pytest.approx(0.0)
    assert metrics["neural"]["dfr"] == pytest.approx(0.0)
    assert metrics["neural"]["precision"] == pytest.approx(4 / 4)
    assert metrics["neural"]["recall"] == pytest.approx(4 / 5)
    assert metrics["neural"]["mask_covers_all"] == pytest.approx(1 / 2)


# edge case


def test_finalize_zero_samples_does_not_crash():
    """num_samples=0 should not crash even with empty stats."""
    stats = init_stats(["d"])
    with pytest.raises(ZeroDivisionError):
        finalize_stats(stats, num_samples=0)
