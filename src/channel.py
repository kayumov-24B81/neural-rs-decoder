import random

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
