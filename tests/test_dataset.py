import numpy as np
import pytest

from src.channel import make_ge_channel
from src.codec import K, N
from src.dataset import RSPositionDataset

INPUT_SIZE = 511  # syndrome bits (256) + zero mask (255)


# HELPERS


def fixed_error_channel(positions):
    """Channel stub that always corrupts the given symbol positions.

    Returns a channel_fn with the standard (noisy, erasures, errors)
    signature, so it is a drop-in replacement for real channels in tests.
    """

    def channel_fn(codeword):
        noisy = bytearray(codeword)
        for p in positions:
            noisy[p] ^= 0xFF
        return bytes(noisy), [], list(positions)

    return channel_fn


# ON-THE-FLY MODE


def test_onthefly_yields_different_samples():
    """Without fixed=True, repeated access regenerates fresh data."""
    ds = RSPositionDataset(size=10, channel_fns=[make_ge_channel("moderate")])
    x1, _ = ds[0]
    x2, _ = ds[0]
    assert not np.array_equal(x1, x2)


def test_onthefly_ignores_index():
    """In on-the-fly mode idx is irrelevant; different idx still works."""
    ds = RSPositionDataset(size=10, channel_fns=[make_ge_channel("moderate")])
    x_a, _ = ds[0]
    x_b, _ = ds[7]
    # both are valid samples of the right shape; values are unrelated to idx
    assert x_a.shape == (INPUT_SIZE,)
    assert x_b.shape == (INPUT_SIZE,)


# FIXED MODE


def test_fixed_yields_identical_samples():
    """With fixed=True, repeated access returns the cached sample."""
    ds = RSPositionDataset(size=10, channel_fns=[make_ge_channel("moderate")], fixed=True)
    x1, y1 = ds[0]
    x2, y2 = ds[0]
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)


def test_fixed_distinct_indices_differ():
    """Different cached indices hold different samples."""
    ds = RSPositionDataset(size=10, channel_fns=[make_ge_channel("moderate")], fixed=True)
    x0, _ = ds[0]
    x1, _ = ds[1]
    assert not np.array_equal(x0, x1)


# SHAPES AND TYPES


def test_sample_shapes_and_dtypes():
    """Input is (511,) float32, positions is (255,) float32."""
    ds = RSPositionDataset(size=5, channel_fns=[make_ge_channel("moderate")])
    x, y = ds[0]
    assert x.shape == (INPUT_SIZE,)
    assert y.shape == (N,)
    assert x.dtype == np.float32
    assert y.dtype == np.float32


def test_positions_are_binary():
    """positions_vector contains only 0.0 and 1.0."""
    ds = RSPositionDataset(size=5, channel_fns=[make_ge_channel("heavy")])
    _, y = ds[0]
    assert set(np.unique(y)).issubset({0.0, 1.0})


def test_len_matches_size():
    for size in (1, 10, 100):
        ds = RSPositionDataset(size=size, channel_fns=[make_ge_channel("light")])
        assert len(ds) == size


# get_raw


def test_get_raw_requires_fixed():
    """get_raw() on a non-fixed dataset raises RuntimeError."""
    ds = RSPositionDataset(size=5, channel_fns=[make_ge_channel("moderate")])
    with pytest.raises(RuntimeError, match="fixed=True"):
        ds.get_raw(0)


def test_get_raw_returns_full_tuple():
    """get_raw() returns (input, positions, noisy, msg) with correct types."""
    ds = RSPositionDataset(size=5, channel_fns=[make_ge_channel("moderate")], fixed=True)
    input_vec, pos_vec, noisy, msg = ds.get_raw(0)
    assert input_vec.shape == (INPUT_SIZE,)
    assert pos_vec.shape == (N,)
    assert len(noisy) == N
    assert len(msg) == K


def test_get_raw_consistent_with_getitem():
    """get_raw()'s first two fields match what __getitem__ returns."""
    ds = RSPositionDataset(size=5, channel_fns=[make_ge_channel("moderate")], fixed=True)
    x, y = ds[2]
    raw_x, raw_y, _, _ = ds.get_raw(2)
    assert np.array_equal(x, raw_x)
    assert np.array_equal(y, raw_y)


# POSITIONS CORRECTNESS (stub channel)


def test_positions_match_corrupted_symbols():
    """positions_vector marks exactly the symbols the channel corrupted."""
    corrupted = [3, 17, 88, 200]
    ds = RSPositionDataset(size=5, channel_fns=[fixed_error_channel(corrupted)], fixed=True)
    _, pos_vec, _, _ = ds.get_raw(0)
    marked = sorted(int(i) for i in np.where(pos_vec > 0.5)[0])
    assert marked == sorted(corrupted)


def test_positions_match_noisy_diff():
    """positions_vector ones align with noisy != codeword (via get_raw)."""
    corrupted = [10, 50, 123]
    ds = RSPositionDataset(size=5, channel_fns=[fixed_error_channel(corrupted)], fixed=True)
    input_vec, pos_vec, noisy, msg = ds.get_raw(0)
    # reconstruct codeword from msg to compare against noisy
    from src.codec import encode

    codeword = encode(msg)
    diff = {i for i in range(N) if noisy[i] != codeword[i]}
    marked = {int(i) for i in np.where(pos_vec > 0.5)[0]}
    assert marked == diff


def test_clean_channel_gives_zero_positions():
    """A channel that corrupts nothing yields an all-zero positions vector."""
    ds = RSPositionDataset(size=5, channel_fns=[fixed_error_channel([])], fixed=True)
    _, pos_vec, _, _ = ds.get_raw(0)
    assert pos_vec.sum() == 0.0


# MIXED CHANNEL


def test_mixed_channel_spans_error_range():
    """A mixed light+heavy dataset produces both light and heavy blocks."""
    np.random.seed(42)
    ds = RSPositionDataset(
        size=1000,
        channel_fns=[make_ge_channel("light"), make_ge_channel("heavy")],
        fixed=True,
    )
    err_counts = np.array([ds.get_raw(i)[1].sum() for i in range(1000)])
    # light averages ~7 errors/block, heavy ~28; a real mix must show spread
    assert err_counts.min() < 15
    assert err_counts.max() > 35
    assert err_counts.std() > 5


def test_single_channel_list_works():
    """A one-element channel list is accepted (no mixing)."""
    ds = RSPositionDataset(size=10, channel_fns=[make_ge_channel("moderate")])
    x, y = ds[0]
    assert x.shape == (INPUT_SIZE,)
