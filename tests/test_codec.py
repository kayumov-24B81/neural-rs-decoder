import os
import random

import numpy as np

from src.codec import ClassicDecoder, OracleDecoder, encode
from src.utils import compute_syndrome_bits

N = 255
K = 223
NSYM = N - K
T = NSYM // 2


def test_encode_decode_no_errors():
    """Encode then decode without errors returns original."""
    msg = os.urandom(K)
    codeword = encode(msg)
    decoded = ClassicDecoder().decode(bytearray(codeword))
    assert decoded == msg


def test_decode_corrects_up_to_t_errors():
    """BM corrects up to 16 symbol errors."""
    msg = os.urandom(K)
    codeword = bytearray(encode(msg))
    positions = random.sample(range(N), T)
    for p in positions:
        codeword[p] ^= random.randint(1, 255)
    decoded = ClassicDecoder().decode(codeword)
    assert decoded == msg


def test_oracle_corrects_up_to_2t():
    """Oracle corrects up to 32 errors via erasures."""
    msg = os.urandom(K)
    codeword = bytearray(encode(msg))
    positions = random.sample(range(N), 2 * T)
    for p in positions:
        codeword[p] ^= random.randint(1, 255)
    decoded = OracleDecoder().decode(codeword, original=encode(msg))
    assert decoded == msg


def test_syndrome_zero_for_valid_codeword():
    """Syndrome of uncorrupted codeword is all zeros."""
    msg = os.urandom(K)
    codeword = encode(msg)
    syndrome = compute_syndrome_bits(codeword)
    assert np.all(syndrome == 0)


def test_syndrome_nonzero_for_corrupted():
    """Syndrome of corrupted codeword is non-zero."""
    msg = os.urandom(K)
    codeword = bytearray(encode(msg))
    codeword[0] ^= 1
    syndrome = compute_syndrome_bits(codeword)
    assert np.any(syndrome != 0)
