import random

import numpy as np

GE_PRESETS = {
    "light": {"p": 2e-3, "r": 0.10, "h": 0.0, "k": 0.5},
    "moderate": {"p": 4e-3, "r": 0.08, "h": 0.0, "k": 0.5},
    "heavy": {"p": 5e-3, "r": 0.05, "h": 0.0, "k": 0.5},
}


def gilbert_elliott_channel(codeword, p, r, h=0.0, k=0.5):
    noisy = bytearray(codeword)
    n = len(codeword)
    symbol_corrupted = [False] * n
    state = 0  # 0 = good, 1 = bad

    for bit_idx in range(n * 8):
        err_prob = h if state == 0 else k
        if random.random() < err_prob:
            sym_idx = bit_idx // 8
            bit_in_sym = 7 - (bit_idx % 8)
            noisy[sym_idx] ^= 1 << bit_in_sym
            symbol_corrupted[sym_idx] = True

        if state == 0:
            if random.random() < p:
                state = 1

        else:
            if random.random() < r:
                state = 0

    errors = [i for i, c in enumerate(symbol_corrupted) if c]
    return bytes(noisy), [], errors


def make_ge_channel(preset="moderate", **overrides):
    if preset == "custom":
        params = overrides
    else:
        if preset not in GE_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. " f"Available: {list(GE_PRESETS)} or 'custom'."
            )
        params = {**GE_PRESETS[preset], **overrides}

    def channel_fn(codeword):
        return gilbert_elliott_channel(codeword, **params)

    return channel_fn


def erasure_channel(codeword, p_erase):
    noisy = bytearray(codeword)
    erasures = []
    for i in range(len(noisy)):
        if random.random() < p_erase:
            noisy[i] = 0
            erasures.append(i)
    return bytes(noisy), erasures, []


def awgn_channel(codeword, ebn0_db):
    """AWGN channel with BPSK modulation and hard decision."""
    n = len(codeword)
    R = 223 / 255
    m = 8
    ebn0_linear = 10 ** (ebn0_db / 10)
    esn0_linear = ebn0_linear * R * m
    sigma = 1.0 / np.sqrt(2 * esn0_linear)

    bits = np.unpackbits(np.frombuffer(codeword, dtype=np.uint8))
    symbols = 1.0 - 2.0 * bits.astype(np.float64)
    noise = np.random.normal(0.0, sigma, len(symbols))
    received = symbols + noise
    decoded_bits = (received < 0).astype(np.uint8)
    noisy_bytes = np.packbits(decoded_bits).tobytes()

    flipped = bits ^ decoded_bits
    symbol_corrupted = flipped.reshape(n, 8).any(axis=1)
    errors = [i for i in range(n) if symbol_corrupted[i]]

    return noisy_bytes, [], errors


AWGN_PRESETS = {
    "light": {"ebn0_db": -3.1},  # FER ~ 0.008
    "moderate": {"ebn0_db": -3.6},  # FER ~ 0.19
    "heavy": {"ebn0_db": -4.2},  # FER ~ 0.83
}


def make_awgn_channel(preset="moderate", **overrides):
    if preset == "custom":
        params = overrides
    else:
        if preset not in AWGN_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. " f"Available: {list(AWGN_PRESETS)} or 'custom'."
            )
        params = {**AWGN_PRESETS[preset], **overrides}

    def channel_fn(codeword):
        return awgn_channel(codeword, **params)

    return channel_fn
