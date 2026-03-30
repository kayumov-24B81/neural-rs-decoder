import os

from src.channel import qsc_erasure_channel
from src.codec import encode

N = 255
K = 223
NSYM = N - K
T = NSYM // 2


def test_qsc_error_rate():
    """QSC produces approximately expected error rate and erasure rate."""

    p_err = 0.1
    p_erase = 0.1
    n_samples = 1000

    erasures = 0
    errors = 0

    for _ in range(n_samples):

        msg = os.urandom(K)
        codeword = encode(msg)
        _, erase_pos, error_pos = qsc_erasure_channel(codeword, p_err, p_erase)

        erasures += len(erase_pos)
        errors += len(error_pos)

    erasure_rate = erasures / (n_samples * N)
    error_rate = errors / (n_samples * N)

    assert abs(error_rate - p_err) < 0.001
    assert abs(erasure_rate - p_erase) < 0.001
