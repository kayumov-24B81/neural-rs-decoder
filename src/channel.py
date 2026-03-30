import random


def qsc_erasure_channel(codeword: bytes, p_err: float, p_erase: float):
    """Simulates a QSC channel with erasures."""
    noisy = bytearray(codeword)
    erasures = []

    for i in range(len(noisy)):
        r = random.random()
        if r < p_erase:
            noisy[i] = 0
            erasures.append(i)
        elif r < p_erase + p_err:
            offset = random.randint(1, 254)
            noisy[i] = ((codeword[i] + offset) % 255) + 1  # ≠ original, ≠ 0

    return bytes(noisy), erasures
