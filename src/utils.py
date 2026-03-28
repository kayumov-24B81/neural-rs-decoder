import numpy as np
from reedsolo import rs_calc_syndromes


def bytes_to_bits(data: bytes):
    """Convert a byte sequence into a bit vector."""
    arr = np.array(list(data), dtype=np.uint8)
    return np.unpackbits(arr)


def bits_to_bytes(data: np.ndarray):
    """Convert a bit vector into a byte sequence."""
    return bytes(np.packbits(data))


def get_zero_mask(data: bytes):
    """Create a mask indicating zero-valued symbols."""
    return np.array([1.0 if b == 0 else 0.0 for b in data], dtype=np.float32)


def compute_syndrome_bits(codeword: bytearray, nsym=32):
    """Compute syndrome of a codeword and convert into a bit vector."""
    syndrome = rs_calc_syndromes(codeword, nsym)[1:]
    return bytes_to_bits(bytes(syndrome))


def build_input(noisy: bytes, nsym=32):
    syndrome_bits = compute_syndrome_bits(noisy, nsym)
    zero_mask = get_zero_mask(noisy)
    return np.concatenate([syndrome_bits, zero_mask])


# TODO: build input with soft information: syndrome + mask + LLR
