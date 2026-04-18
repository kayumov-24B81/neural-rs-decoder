import random

import pytest

from src.channel import GE_PRESETS, erasure_channel, gilbert_elliott_channel, make_ge_channel
from src.codec import K, N, encode


@pytest.fixture
def codeword():
    random.seed(0)
    msg = bytes(random.randint(0, 255) for _ in range(K))
    return encode(msg)


def test_ge_zero_channel_no_errors(codeword):
    random.seed(42)
    noisy, erasures, errors = gilbert_elliott_channel(codeword, p=0.1, r=0.1, h=0.0, k=0.0)
    assert noisy == bytes(codeword)
    assert erasures == []
    assert errors == []


def test_ge_full_channel_all_corrupted(codeword):
    random.seed(42)
    noisy, _, errors = gilbert_elliott_channel(codeword, p=0.1, r=0.1, h=1.0, k=1.0)
    assert len(errors) == N
    assert noisy != bytes(codeword)


def test_ge_returns_empty_erasures(codeword):
    random.seed(42)
    _, erasures, _ = gilbert_elliott_channel(codeword, p=0.01, r=0.1, h=0.0, k=0.5)
    assert erasures == []


def test_ge_errors_match_diff(codeword):
    random.seed(42)
    noisy, _, errors = gilbert_elliott_channel(codeword, p=0.01, r=0.1, h=0.0, k=0.5)
    expected = {i for i in range(N) if noisy[i] != codeword[i]}
    assert set(errors) == expected


def test_ge_stationary_ber():
    random.seed(42)
    p, r, h, k = 0.05, 0.1, 0.0, 0.5
    # pi_B = p / (p + r), avg_ber = pi_G * h + pi_B * k
    expected_ber = (p / (p + r)) * k + (r / (p + r)) * h

    n_samples = 2000
    total_bits = 0
    flipped_bits = 0
    msg = bytes(K)
    cw = encode(msg)

    for _ in range(n_samples):
        noisy, _, _ = gilbert_elliott_channel(cw, p=p, r=r, h=h, k=k)
        for orig_b, noisy_b in zip(cw, noisy):
            diff = orig_b ^ noisy_b
            flipped_bits += bin(diff).count("1")
            total_bits += 8

    observed_ber = flipped_bits / total_bits
    assert abs(observed_ber - expected_ber) < 0.01


def test_ge_produces_bursts():
    random.seed(42)
    p, r, h, k = 0.02, 0.1, 0.0, 0.5  # avg BER ~ 0.083

    n_samples = 500
    msg = bytes(K)
    cw = encode(msg)

    max_run_lengths = []
    for _ in range(n_samples):
        noisy, _, _ = gilbert_elliott_channel(cw, p=p, r=r, h=h, k=k)
        run = 0
        best = 0
        for i in range(N):
            if noisy[i] != cw[i]:
                run += 1
                best = max(best, run)
            else:
                run = 0
        max_run_lengths.append(best)

    mean_max_run = sum(max_run_lengths) / n_samples
    assert mean_max_run > 5


def test_ge_presets_exist():
    required_keys = {"p", "r", "h", "k"}
    for name in ("light", "moderate", "heavy"):
        assert name in GE_PRESETS
        assert set(GE_PRESETS[name]) == required_keys


def test_make_ge_channel_by_preset(codeword):
    for name in ("light", "moderate", "heavy"):
        fn = make_ge_channel(name)
        random.seed(42)
        noisy, erasures, errors = fn(codeword)
        assert len(noisy) == N
        assert erasures == []
        assert isinstance(errors, list)


def test_make_ge_channel_custom(codeword):
    fn = make_ge_channel("custom", p=0.01, r=0.1, h=0.0, k=0.5)
    random.seed(42)
    noisy, _, _ = fn(codeword)
    assert len(noisy) == N


def test_make_ge_channel_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        make_ge_channel("unknown_preset")


def test_erasure_channel_rate(codeword):
    random.seed(42)
    p_erase = 0.1
    n_samples = 1000

    total_erasures = 0
    for _ in range(n_samples):
        _, erasures, _ = erasure_channel(codeword, p_erase)
        total_erasures += len(erasures)

    observed_rate = total_erasures / (n_samples * N)
    assert abs(observed_rate - p_erase) < 0.005


def test_erase_positions_zeroed(codeword):
    random.seed(42)
    noisy, erasures, _ = erasure_channel(codeword, p_erase=0.2)
    for pos in erasures:
        assert noisy[pos] == 0


def test_erasure_returns_empty_errors(codeword):
    random.seed(42)
    _, _, errors = erasure_channel(codeword, p_erase=0.2)
    assert errors == []
